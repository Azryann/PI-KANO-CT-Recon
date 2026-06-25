import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time

from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from physics import RadonPhysics

class PAUM_Surrogate(nn.Module):
    def __init__(self, img_size, num_angles, num_detectors, num_cascades=3, device='cuda'):
        super().__init__()
        self.num_cascades = num_cascades
        self.physics = RadonPhysics(img_size, num_angles, num_detectors, device=device)
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 1, 3, padding=1)
            ) for _ in range(num_cascades)
        ])

    def forward(self, y):
        tau_safe = torch.clamp(self.tau, min=1e-4, max=2.0)
        x = self.physics.adjoint(y) * tau_safe
        for i in range(self.num_cascades):
            grad = self.physics.adjoint(self.physics.forward(x) - y) * tau_safe
            update = self.blocks[i](torch.cat([x, grad], dim=1))
            x = x - update
        return x

class JotlasNet_Surrogate(nn.Module):
    def __init__(self, img_size, num_angles, num_detectors, num_cascades=2, device='cuda'):
        super().__init__()
        self.num_cascades = num_cascades
        self.physics = RadonPhysics(img_size, num_angles, num_detectors, device=device)
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.embed = nn.Conv2d(2, 64, kernel_size=4, stride=4) 
        self.transformer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, batch_first=True)
        self.de_embed = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4)

    def forward(self, y):
        tau_safe = torch.clamp(self.tau, min=1e-4, max=2.0)
        x = self.physics.adjoint(y) * tau_safe
        B, C_orig, H_orig, W_orig = x.shape
        for _ in range(self.num_cascades):
            grad = self.physics.adjoint(self.physics.forward(x) - y) * tau_safe
            feat = self.embed(torch.cat([x, grad], dim=1))
            B_f, C_f, H_f, W_f = feat.shape
            feat_flat = feat.view(B_f, C_f, -1).permute(0, 2, 1)
            feat_trans = self.transformer(feat_flat).permute(0, 2, 1).view(B_f, C_f, H_f, W_f)
            update = self.de_embed(feat_trans)
            if update.shape[-2:] != (H_orig, W_orig):
                update = F.interpolate(update, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            x = x - update
        return x

def compute_metrics(pred, gt):
    pred_np = pred.detach().cpu().squeeze().numpy()
    gt_np = gt.detach().cpu().squeeze().numpy()
    if len(pred_np.shape) == 2:
        pred_np, gt_np = [pred_np], [gt_np]
    psnrs, ssims = [], []
    for p, g in zip(pred_np, gt_np):
        rmse = np.sqrt(np.mean((p - g) ** 2))
        data_range = g.max() - g.min() + 1e-8
        psnrs.append(20 * np.log10(data_range / (rmse + 1e-8)))
        ssims.append(ssim(p, g, data_range=data_range, win_size=3))
    return np.mean(psnrs), np.mean(ssims)

def train_and_evaluate(model_name, dataset_name, data_path, epochs=5, batch_size=2, device='cuda'):
    print(f"\n{'='*50}\nStarting Benchmark: {model_name} on {dataset_name.upper()}\n{'='*50}")
    
    img_size, angles, detectors, phys_scale = (362, 1000, 513, 0.1) if dataset_name == 'lodopab' else (512, 736, 736, 0.2)
    dataloader = get_ct_dataloader(dataset_name, data_path, batch_size=batch_size)
    
    if model_name == 'PI_KINN': model = PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    elif model_name == 'PAUM': model = PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    elif model_name == 'JotlasNet': model = JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # Q1 UPGRADE: Ensure T_max matches EXACTLY the number of epochs run, 
    # so the learning rate hits 1e-6 at the final step, guaranteeing convergence.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    start_epoch = 0
    checkpoints = glob.glob(f"{model_name}_checkpoint_ep*.pth")
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print(f"Found checkpoint: {latest_ckpt}. Resuming training...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
    
    if start_epoch >= epochs:
        print(f"Model {model_name} has already completed {epochs} epochs. Skipping.")
        return

    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, epoch_psnr, epoch_ssim = 0.0, 0.0, 0.0
        steps = 0
        start_time = time.time()
        
        for batch_idx, (sinograms, ground_truths) in enumerate(dataloader):
            sinograms, ground_truths = sinograms.to(device) / phys_scale, ground_truths.to(device) / phys_scale
            optimizer.zero_grad()
            
            reconstructions = model(sinograms)
            
            # FIX: Reverted to pure MSE to mathematically guarantee PSNR maximization
            alignment_loss = F.mse_loss(reconstructions, ground_truths)
            fidelity_loss = F.mse_loss(model.physics(reconstructions), sinograms)
            loss = alignment_loss + (1e-4 * fidelity_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            psnr_val, ssim_val = compute_metrics(reconstructions, ground_truths)
            epoch_loss += loss.item()
            epoch_psnr += psnr_val
            epoch_ssim += ssim_val
            steps += 1
            
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Ep [{epoch+1}/{epochs}] Stp [{batch_idx}] | LR: {current_lr:.2e} | Loss: {loss.item():.4f} | PSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f}")
                
        scheduler.step()
        torch.save({
            'epoch': epoch + 1, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
        }, f"{model_name}_checkpoint_ep{epoch+1}.pth")
        
        print(f"\n--- {model_name} Epoch {epoch+1} Summary ---")
        print(f"Time: {time.time() - start_time:.1f}s | Avg PSNR: {epoch_psnr/steps:.2f}dB | Avg SSIM: {epoch_ssim/steps:.4f}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    
    if os.path.exists(LODOPAB_PATH):
        print("Kaggle environment detected. Commencing SOTA Benchmarking on LoDoPaB-CT...")
        
        # We commented out the baselines to prioritize our proposed model!
        # train_and_evaluate('PAUM', 'lodopab', LODOPAB_PATH, epochs=50, batch_size=2, device=device)
        # train_and_evaluate('JotlasNet', 'lodopab', LODOPAB_PATH, epochs=50, batch_size=2, device=device)
        
        # Run PI-KINN immediately
        train_and_evaluate('PI_KINN', 'lodopab', LODOPAB_PATH, epochs=5, batch_size=2, device=device)
    else:
        train_and_evaluate('PI_KINN', 'lodopab', 'dummy_path.tfrecord', epochs=1, batch_size=1, device=device)