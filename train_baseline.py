import os
import glob
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pi_kinn import KirchhoffPhysicsConstraint
# Import our custom modules
from dataloaders import get_ct_dataloader
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
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
    p_mu_np = pred_mu.detach().cpu().numpy()
    g_mu_np = gt_mu.detach().cpu().numpy()
    
    pred_hu_norm = apply_clinical_window(mu_to_hu(pred_mu)).detach().cpu().numpy()
    gt_hu_norm = apply_clinical_window(mu_to_hu(gt_mu)).detach().cpu().numpy()
    
    FIXED_MU_RANGE = 0.04 
    psnr_mu_list, psnr_hu_list, ssim_list = [], [], []
    
    for i in range(p_mu_np.shape[0]):
        p_m, g_m = p_mu_np[i, 0], g_mu_np[i, 0]
        p_h, g_h = pred_hu_norm[i, 0], gt_hu_norm[i, 0]
        
        mse_mu = np.mean((p_m - g_m)**2)
        psnr_mu_list.append(10 * np.log10((FIXED_MU_RANGE**2) / (mse_mu + 1e-8)))
        
        mse_hu = np.mean((p_h - g_h)**2)
        psnr_hu_list.append(10 * np.log10(1.0 / (mse_hu + 1e-8)))
        ssim_list.append(ssim(p_h, g_h, data_range=1.0))
        
    return np.mean(psnr_mu_list), np.mean(psnr_hu_list), np.mean(ssim_list)

# ==========================================
# Q1 TRAINING PIPELINE (60 Epochs, 20% Subset)
# ==========================================
def train_baseline_subset(model_name, data_path, val_path=None, device='cuda'):
    print(f"\n{'='*75}\nDAY 2 LAUNCH: 60-Epoch {model_name} (20% Subset + Q1 Rigor)\n{'='*75}")
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    batch_size = 2
    epochs = 60
    steps_per_epoch = 3582  # 20% of LoDoPaB
    
    train_loader = get_ct_dataloader('lodopab', data_path, batch_size=batch_size)
    val_loader = get_ct_dataloader('lodopab', val_path, batch_size=batch_size) if val_path and os.path.exists(val_path) else train_loader
    
    # Initialize the requested baseline
    if model_name == 'PAUM':
        model = PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    elif model_name == 'JotlasNet':
        model = JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device)
    else:
        raise ValueError("Invalid model name.")
    
    kirchhoff_op = KirchhoffPhysicsConstraint(img_size, angles, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # --- RESUME LOGIC ---
    start_epoch = 0
    best_val_hu_psnr = -float('inf')
    checkpoints = glob.glob(f"{model_name}_subset_ep*.pth")
    
    if checkpoints:
        def get_epoch_num(fpath):
            try: return int(os.path.basename(fpath).split('_ep')[1].split('.pth')[0])
            except ValueError: return -1
                
        latest_ckpt = max(checkpoints, key=get_epoch_num)
        print(f"Found checkpoint: {latest_ckpt}. Resuming training...")
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        best_val_hu_psnr = checkpoint.get('best_val_psnr', -float('inf'))
        print(f"Successfully resumed from Epoch {start_epoch}.")
        
    if start_epoch >= epochs:
        print(f"Training already completed {epochs} epochs.")
        return

    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        metrics = {'loss': 0.0, 'l_mu': 0.0, 'l_hu': 0.0, 'l_phys': 0.0}
        start_time = time.time()
        
        train_iter = iter(train_loader)
        
        # ---------------------------------------------------------
        # 1. TRAINING PHASE
        # ---------------------------------------------------------
        for step in range(steps_per_epoch):
            try:
                sinograms, ground_truths = next(train_iter)
            except StopIteration:
                break
                
            sinograms = sinograms.to(device) / phys_scale
            ground_truths = ground_truths.to(device) / phys_scale
            
            optimizer.zero_grad()
            reconstructions = model(sinograms)
            
            # Loss 1: mu-space MSE (Structure)
            loss_mu = F.mse_loss(reconstructions, ground_truths)
            
            # Loss 2: HU-space L1 (Kills DC Bias)
            pred_hu_scaled = mu_to_hu(reconstructions * phys_scale) / 1000.0
            gt_hu_scaled = mu_to_hu(ground_truths * phys_scale) / 1000.0
            loss_hu = F.l1_loss(pred_hu_scaled, gt_hu_scaled)
            
            # Loss 3: Standard Physics Fidelity (Baselines use standard Radon physics)
            # Loss 3: Scale-Invariant Physics Constraint (Identical to PI-KINN)
            k_out = kirchhoff_op(reconstructions)
            target_k = sinograms.mean(dim=-1)
            loss_phys = 1.0 - F.cosine_similarity(k_out.flatten(1), target_k.flatten(1)).mean() 
            
            # Composite Loss (Identical to PI-KINN)
            loss = loss_mu + loss_hu + (0.1 * loss_phys)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            metrics['loss'] += loss.item(); metrics['l_mu'] += loss_mu.item()
            metrics['l_hu'] += loss_hu.item(); metrics['l_phys'] += loss_phys.item()
            
            if step % 500 == 0:
                print(f"Ep [{epoch+1}/{epochs}] Stp [{step}/{steps_per_epoch}] | Tot: {loss.item():.4f} | μ-MSE: {loss_mu.item():.4f} | HU-L1: {loss_hu.item():.4f} | Phys: {loss_phys.item():.4f}")

        # ---------------------------------------------------------
        # 2. VALIDATION PHASE
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
        
        # ---------------------------------------------------------
        # 3. CHECKPOINTING
        # ---------------------------------------------------------
        ckpt_data = {
            'epoch': epoch + 1, 
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 
            'scheduler_state': scheduler.state_dict(),
            'best_val_psnr': best_val_hu_psnr
        }
        
        torch.save(ckpt_data, f"{model_name}_subset_ep{epoch+1}.pth")
        
        is_best = ""
        if avg_p_hu > best_val_hu_psnr:
            best_val_hu_psnr = avg_p_hu
            ckpt_data['best_val_psnr'] = best_val_hu_psnr
            torch.save(ckpt_data, f"{model_name}_subset_BEST.pth")
            is_best = "★ NEW BEST"
            
        print(f"\n{'-'*75}")
        print(f"EPOCH {epoch+1} SUMMARY | Time: {time.time() - start_time:.1f}s | LR: {current_lr:.2e}")
        print(f"TRAIN LOSSES -> Tot: {metrics['loss']/steps_per_epoch:.4f} | μ-MSE: {metrics['l_mu']/steps_per_epoch:.4f} | HU-L1: {metrics['l_hu']/steps_per_epoch:.4f}")
        print(f"VALIDATION   -> μ-PSNR: {avg_p_mu:.2f} dB | HU-PSNR: {avg_p_hu:.2f} dB | SSIM: {avg_ssim:.4f} {is_best}")
        print(f"{'-'*75}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    VAL_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_validation.tfrecord" 
    
    if os.path.exists(TRAIN_PATH):
        # CHANGE THIS VARIABLE TO 'JotlasNet' TO TRAIN THE TRANSFORMER
        MODEL_TO_TRAIN = 'PAUM' 
        train_baseline_subset(MODEL_TO_TRAIN, TRAIN_PATH, val_path=VAL_PATH, device=device)