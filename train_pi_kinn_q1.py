import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Import our custom modules
from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from physics import RadonPhysics

# ==========================================
# Q1 CLINICAL MATH FUNCTIONS
# ==========================================
def mu_to_hu(mu_tensor):
    mu_water = 0.0192
    return 1000.0 * (mu_tensor - mu_water) / mu_water

def apply_clinical_window(hu_tensor):
    # Strict Q1 clinical lung window: [-1000, 400] HU
    clipped = torch.clamp(hu_tensor, min=-1000.0, max=400.0)
    return (clipped + 1000.0) / 1400.0

def compute_hu_psnr(pred_mu, gt_mu):
    """ Computes PSNR strictly in the windowed HU domain for validation tracking. """
    pred_norm = apply_clinical_window(mu_to_hu(pred_mu))
    gt_norm = apply_clinical_window(mu_to_hu(gt_mu))
    
    mse = F.mse_loss(pred_norm, gt_norm).item()
    if mse < 1e-12: return 100.0
    return 10 * np.log10(1.0 / mse)

# ==========================================
# Q1 TRAINING PIPELINE
# ==========================================
def train_pi_kinn(data_path, epochs=15, batch_size=2, device='cuda'):
    print(f"\n{'='*60}\nStarting Q1-Standard Training: PI-KINN\n{'='*60}")
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    
    # We use the same dataloader, but we will split the stream: 
    # First N steps for training, next 50 steps for validation.
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=batch_size)
    
    model = PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_psnr = -float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        start_time = time.time()
        
        data_iterator = iter(dataloader)
        
        # ---------------------------------------------------------
        # 1. TRAINING PHASE (e.g., 2000 steps per epoch to ensure we can do many epochs)
        # ---------------------------------------------------------
        train_steps_per_epoch = 2000 
        
        for _ in range(train_steps_per_epoch):
            try:
                sinograms, ground_truths = next(data_iterator)
            except StopIteration:
                break # End of dataset
                
            sinograms = sinograms.to(device) / phys_scale
            ground_truths = ground_truths.to(device) / phys_scale
            
            optimizer.zero_grad()
            reconstructions = model(sinograms)
            
            # --- Q1 DUAL-OBJECTIVE LOSS ---
            # Loss 1: mu-space MSE (Global Structure)
            loss_mu = F.mse_loss(reconstructions, ground_truths)
            
            # Loss 2: HU-space Windowed MSE (Forces exact clinical intensity scaling)
            pred_hu_norm = apply_clinical_window(mu_to_hu(reconstructions * phys_scale))
            gt_hu_norm = apply_clinical_window(mu_to_hu(ground_truths * phys_scale))
            loss_hu = F.mse_loss(pred_hu_norm, gt_hu_norm)
            
            # Loss 3: Physics Fidelity
            loss_phys = F.mse_loss(model.physics(reconstructions), sinograms)
            
            # Total Loss (HU loss is weighted heavily to kill the DC bias)
            loss = loss_mu + (1.0 * loss_hu) + (1e-4 * loss_phys)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
            if steps % 100 == 0:
                print(f"Ep [{epoch+1}/{epochs}] Train Stp [{steps}] | Loss: {loss.item():.4f} | mu-MSE: {loss_mu.item():.4f} | HU-MSE: {loss_hu.item():.4f}")

        # ---------------------------------------------------------
        # 2. VALIDATION PHASE (50 steps)
        # ---------------------------------------------------------
        model.eval()
        val_psnr_total = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for _ in range(50):
                try:
                    sinograms, ground_truths = next(data_iterator)
                except StopIteration:
                    break
                    
                sinograms = sinograms.to(device) / phys_scale
                ground_truths = ground_truths.to(device) / phys_scale
                
                reconstructions = model(sinograms)
                
                # Calculate strict HU-Windowed PSNR
                psnr = compute_hu_psnr(reconstructions * phys_scale, ground_truths * phys_scale)
                val_psnr_total += psnr
                val_steps += 1
                
        avg_val_psnr = val_psnr_total / max(1, val_steps)
        scheduler.step()
        
        # ---------------------------------------------------------
        # 3. CHECKPOINTING (Save Best Model)
        # ---------------------------------------------------------
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_psnr': avg_val_psnr
        }
        
        # Save current epoch
        torch.save(checkpoint_data, f"PI_KINN_ep{epoch+1}.pth")
        
        # Save BEST epoch
        is_best = ""
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save(checkpoint_data, "PI_KINN_BEST.pth")
            is_best = "--> [NEW BEST MODEL SAVED]"
            
        epoch_time = time.time() - start_time
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Time: {epoch_time:.1f}s | Train Loss: {epoch_loss/steps:.4f} | Val HU-PSNR: {avg_val_psnr:.2f} dB {is_best}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    
    if os.path.exists(LODOPAB_PATH):
        # Train for 15 epochs. With 2000 steps per epoch, this will take ~10 hours total.
        train_pi_kinn(LODOPAB_PATH, epochs=15, batch_size=2, device=device)
    else:
        print("Dataset not found.")