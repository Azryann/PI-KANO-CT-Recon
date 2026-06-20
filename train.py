import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Import our custom PI-KANO modules
from dataloaders import get_lodopab_dataloader
from pi_kano import PI_KANO

def compute_metrics(pred, gt):
    pred_np = pred.detach().cpu().squeeze().numpy()
    gt_np = gt.detach().cpu().squeeze().numpy()
    
    if len(pred_np.shape) == 2:
        pred_np = [pred_np]
        gt_np = [gt_np]
        
    psnrs, nrmses, ssims = [], [], []
    
    for p, g in zip(pred_np, gt_np):
        rmse = np.sqrt(np.mean((p - g) ** 2))
        nrmse = rmse / (np.mean(g) + 1e-8)
        nrmses.append(nrmse)
        
        data_range = g.max() - g.min() + 1e-8
        psnr = 20 * np.log10(data_range / (rmse + 1e-8))
        psnrs.append(psnr)
        
        s_val = ssim(p, g, data_range=data_range, win_size=3)
        ssims.append(s_val)
        
    return np.mean(psnrs), np.mean(nrmses), np.mean(ssims)


def train_pi_kano(tfrecord_path, epochs=50, batch_size=2, lr=1e-3, device='cuda'):
    print(f"Setting up PI-KANO Training on {device}...")
    
    dataloader = get_lodopab_dataloader(tfrecord_path, batch_size=batch_size)
    model = PI_KANO(img_size=362, num_angles=1000, num_detectors=513, device=device).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Q1 Journal Standard: Cosine Annealing Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    lambda_physics = 0.1
    model.train()
    print("Beginning Training Loop...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_psnr, epoch_nrmse, epoch_ssim = 0.0, 0.0, 0.0
        steps = 0
        
        for batch_idx, (sinograms, ground_truths) in enumerate(dataloader):
            sinograms = sinograms.to(device)
            ground_truths = ground_truths.to(device)
            
            # 3. PHYSICAL DATA SCALING:
            # LoDoPaB physical attenuation max is ~0.1. 
            # We scale by 0.1 to bring targets into the [0, 1] range.
            PHYSICAL_CONSTANT = 0.1
            sinograms = sinograms / PHYSICAL_CONSTANT
            ground_truths = ground_truths / PHYSICAL_CONSTANT
            
            optimizer.zero_grad()
            
            # Forward pass
            # Forward pass
            reconstructions = model(sinograms)
            
            # 1. Image Alignment Loss
            alignment_loss = F.mse_loss(reconstructions, ground_truths)
            
            # 2. Physics Data Fidelity Loss
            pred_sinograms = model.physics(reconstructions)
            fidelity_loss = F.mse_loss(pred_sinograms, sinograms)
            
            # 3. KAN Curvature Penalty (NEW - Suppresses Spline Oscillations)
            curve_penalty = model.compute_kan_regularization()
            
            # Hyperparameters for Q1 stability
            lambda_physics = 0.0001
            lambda_curve = 0.01  # Forces the B-splines to remain smooth
            
            # Fused Physics-Informed Loss
            total_loss = alignment_loss + (lambda_physics * fidelity_loss) + (lambda_curve * curve_penalty)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient Clipping (Extra safety measure against exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Metrics
            psnr_val, nrmse_val, ssim_val = compute_metrics(reconstructions, ground_truths)
            
            epoch_loss += total_loss.item()
            epoch_psnr += psnr_val
            epoch_nrmse += nrmse_val
            epoch_ssim += ssim_val
            steps += 1
            
            # Print every 50 steps to keep Kaggle logs clean
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}] "
                      f"| LR: {current_lr:.2e} | Loss: {total_loss.item():.6f} "
                      f"| PSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f}")
        
        # Step the scheduler at the end of the epoch
        scheduler.step()
        
        # Epoch Summary
        print(f"\n=== Epoch {epoch+1} Summary ===")
        print(f"Average Loss:  {epoch_loss / steps:.6f}")
        print(f"Average PSNR:  {epoch_psnr / steps:.2f} dB")
        print(f"Average SSIM:  {epoch_ssim / steps:.4f}\n")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"pi_kano_epoch_{epoch+1}.pth")
        print(f"Model checkpoint saved to pi_kano_epoch_{epoch+1}.pth\n")


if __name__ == "__main__":
    KAGGLE_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(KAGGLE_PATH):
        print("Kaggle environment verified. Commencing full-scale training on 85GB LoDoPaB...")
        # 50 Epochs is the standard for convergence on this dataset
        train_pi_kano(KAGGLE_PATH, epochs=50, batch_size=1, lr=1e-3, device=device)
    else:
        print("Kaggle dataset not found. Using local dummy streaming for prototyping...")
        train_pi_kano("dummy_path", epochs=1, batch_size=1, lr=1e-3, device=device)