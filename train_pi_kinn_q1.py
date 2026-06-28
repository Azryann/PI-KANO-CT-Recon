import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from skimage.metrics import structural_similarity as ssim

from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from physics import RadonPhysics

# ==========================================
# Q1 CLINICAL MATH & METRICS
# ==========================================
def mu_to_hu(mu_tensor):
    mu_water = 0.0192
    return 1000.0 * (mu_tensor - mu_water) / mu_water

def apply_clinical_window(hu_tensor):
    clipped = torch.clamp(hu_tensor, min=-1000.0, max=400.0)
    return (clipped + 1000.0) / 1400.0

def compute_comprehensive_metrics(pred_mu, gt_mu):
    p_mu_np = pred_mu.detach().cpu().squeeze().numpy()
    g_mu_np = gt_mu.detach().cpu().squeeze().numpy()
    
    # Fixed Dataset-Wide Dynamic Range for mu-space
    FIXED_MU_RANGE = 0.04 
    mse_mu = np.mean((p_mu_np - g_mu_np)**2)
    psnr_mu = 10 * np.log10((FIXED_MU_RANGE**2) / (mse_mu + 1e-8))
    
    # Windowed HU-space
    pred_hu_norm = apply_clinical_window(mu_to_hu(pred_mu)).detach().cpu().squeeze().numpy()
    gt_hu_norm = apply_clinical_window(mu_to_hu(gt_mu)).detach().cpu().squeeze().numpy()
    
    mse_hu = np.mean((pred_hu_norm - gt_hu_norm)**2)
    psnr_hu = 10 * np.log10(1.0 / (mse_hu + 1e-8))
    ssim_val = ssim(pred_hu_norm, gt_hu_norm, data_range=1.0)
    
    return psnr_mu, psnr_hu, ssim_val

# ==========================================
# Q1 TRAINING PIPELINE (Clean & Ablated)
# ==========================================
def train_pi_kinn(train_path, val_path=None, epochs=15, batch_size=2, device='cuda'):
    print(f"\n{'='*75}\nStarting Q1-Standard Training: PI-KINN (Clean Methodology)\n{'='*75}")
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    
    train_loader = get_ct_dataloader('lodopab', train_path, batch_size=batch_size)
    val_loader = get_ct_dataloader('lodopab', val_path, batch_size=batch_size) if val_path and os.path.exists(val_path) else train_loader
        
    model = PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    best_val_hu_psnr = -float('inf')
    
    for epoch in range(epochs):
        model.train()
        metrics = {'loss': 0.0, 'l_mu': 0.0, 'l_hu': 0.0, 'l_phys': 0.0}
        steps = 0
        start_time = time.time()
        
        train_iter = iter(train_loader)
        train_steps_per_epoch = 2000 
        
        for _ in range(train_steps_per_epoch):
            try:
                sinograms, ground_truths = next(train_iter)
            except StopIteration:
                break
                
            sinograms = sinograms.to(device) / phys_scale
            ground_truths = ground_truths.to(device) / phys_scale
            
            optimizer.zero_grad()
            reconstructions = model(sinograms)
            
            # 1. mu-space MSE (Primary Reconstruction Fidelity)
            loss_mu = F.mse_loss(reconstructions, ground_truths)
            
            # 2. HU-space L1 Loss (Domain Alignment / DC Bias Corrector)
            pred_hu_scaled = mu_to_hu(reconstructions * phys_scale) / 1000.0
            gt_hu_scaled = mu_to_hu(ground_truths * phys_scale) / 1000.0
            loss_hu = F.l1_loss(pred_hu_scaled, gt_hu_scaled)
            
            # 3. Physics Fidelity
            loss_phys = F.mse_loss(model.physics(reconstructions), sinograms)
            
            # Clean, fixed-weight composite loss
            loss = loss_mu + loss_hu + (1e-4 * loss_phys)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            metrics['loss'] += loss.item(); metrics['l_mu'] += loss_mu.item()
            metrics['l_hu'] += loss_hu.item(); metrics['l_phys'] += loss_phys.item()
            steps += 1
            
            if steps % 200 == 0:
                print(f"Ep [{epoch+1}/{epochs}] Stp [{steps}] | Tot: {loss.item():.4f} | μ-MSE: {loss_mu.item():.4f} | HU-L1: {loss_hu.item():.4f}")

        # ---------------------------------------------------------
        # VALIDATION PHASE
        # ---------------------------------------------------------
        model.eval()
        val_metrics = {'p_mu': 0.0, 'p_hu': 0.0, 'ssim': 0.0}
        val_steps = 0
        val_iter = iter(val_loader)
        
        with torch.no_grad():
            for _ in range(50): 
                try:
                    sinograms, ground_truths = next(val_iter)
                except StopIteration:
                    break
                    
                sinograms = sinograms.to(device) / phys_scale
                ground_truths = ground_truths.to(device) / phys_scale
                reconstructions = model(sinograms)
                
                p_mu, p_hu, s_val = compute_comprehensive_metrics(reconstructions * phys_scale, ground_truths * phys_scale)
                val_metrics['p_mu'] += p_mu; val_metrics['p_hu'] += p_hu; val_metrics['ssim'] += s_val
                val_steps += 1
                
        avg_p_mu = val_metrics['p_mu'] / val_steps
        avg_p_hu = val_metrics['p_hu'] / val_steps
        avg_ssim = val_metrics['ssim'] / val_steps
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        ckpt_data = {
            'epoch': epoch + 1, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'val_hu_psnr': avg_p_hu
        }
        torch.save(ckpt_data, f"PI_KINN_ep{epoch+1}.pth")
        
        is_best = ""
        if avg_p_hu > best_val_hu_psnr:
            best_val_hu_psnr = avg_p_hu
            torch.save(ckpt_data, "PI_KINN_BEST.pth")
            is_best = "★ NEW BEST"
            
        print(f"\n{'-'*75}")
        print(f"EPOCH {epoch+1} SUMMARY | Time: {time.time() - start_time:.1f}s | LR: {current_lr:.2e}")
        print(f"TRAIN LOSSES -> Tot: {metrics['loss']/steps:.4f} | μ-MSE: {metrics['l_mu']/steps:.4f} | HU-L1: {metrics['l_hu']/steps:.4f}")
        print(f"VALIDATION   -> μ-PSNR: {avg_p_mu:.2f} dB | HU-PSNR: {avg_p_hu:.2f} dB | SSIM: {avg_ssim:.4f} {is_best}")
        print(f"{'-'*75}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    VAL_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_validation.tfrecord" 
    
    if os.path.exists(TRAIN_PATH):
        train_pi_kinn(TRAIN_PATH, val_path=VAL_PATH, epochs=15, batch_size=2, device=device)