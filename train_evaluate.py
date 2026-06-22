import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time

# Import our custom modules
from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from physics import RadonPhysics

# ==========================================
# SOTA SURROGATE BASELINES (2025)
# ==========================================
class PAUM_Surrogate(nn.Module):
    """ Surrogate for Physics-Aware Unrolled Model (PAUM, 2025). Uses standard CNN cascades. """
    def __init__(self, img_size, num_angles, num_detectors, num_cascades=3, device='cuda'):
        super().__init__()
        self.num_cascades = num_cascades
        self.physics = RadonPhysics(img_size, num_angles, num_detectors, device=device)
        self.tau = nn.Parameter(torch.tensor(1e-3))
        
        # Standard independent CNN blocks for each cascade (No stateful ODE memory)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 1, 3, padding=1)
            ) for _ in range(num_cascades)
        ])

    def forward(self, y):
        x = self.physics.adjoint(y) * self.tau
        for i in range(self.num_cascades):
            grad = self.physics.adjoint(self.physics.forward(x) - y) * self.tau
            update = self.blocks[i](torch.cat([x, grad], dim=1))
            x = x - update
        return x

class JotlasNet_Surrogate(nn.Module):
    """ Surrogate for JotlasNet (2025). Uses a heavy Unrolled Transformer (Memory Intensive). """
    def __init__(self, img_size, num_angles, num_detectors, num_cascades=2, device='cuda'):
        super().__init__()
        self.num_cascades = num_cascades
        self.physics = RadonPhysics(img_size, num_angles, num_detectors, device=device)
        self.tau = nn.Parameter(torch.tensor(1e-3))
        
        # Transformer blocks (Using MultiheadAttention over flattened patches)
        self.embed = nn.Conv2d(2, 64, kernel_size=4, stride=4) # Patch embedding
        self.transformer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, batch_first=True)
        self.de_embed = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4)

    def forward(self, y):
        x = self.physics.adjoint(y) * self.tau
        for _ in range(self.num_cascades):
            grad = self.physics.adjoint(self.physics.forward(x) - y) * self.tau
            feat = self.embed(torch.cat([x, grad], dim=1))
            B, C, H, W = feat.shape
            
            # Flatten for transformer
            feat_flat = feat.view(B, C, -1).permute(0, 2, 1)
            feat_trans = self.transformer(feat_flat)
            feat_trans = feat_trans.permute(0, 2, 1).view(B, C, H, W)
            
            update = self.de_embed(feat_trans)
            x = x - update
        return x

# ==========================================
# TRAINING & EVALUATION PIPELINE
# ==========================================
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

def train_and_evaluate(model_name, dataset_name, data_path, epochs=10, batch_size=2, device='cuda'):
    print(f"\n{'='*50}\nStarting Benchmark: {model_name} on {dataset_name.upper()}\n{'='*50}")
    
    # Dimensions based on dataset
    if dataset_name == 'lodopab':
        img_size, angles, detectors = 362, 1000, 513
        phys_scale = 0.1
    else: # Mayo Clinic
        img_size, angles, detectors = 512, 736, 736
        phys_scale = 0.2  # Typical Mayo normalization
        
    dataloader = get_ct_dataloader(dataset_name, data_path, batch_size=batch_size)
    
    # Initialize requested model
    if model_name == 'PI_KINN':
        model = PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    elif model_name == 'PAUM':
        model = PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    elif model_name == 'JotlasNet':
        model = JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_psnr, epoch_ssim = 0.0, 0.0, 0.0
        steps = 0
        start_time = time.time()
        
        for batch_idx, (sinograms, ground_truths) in enumerate(dataloader):
            sinograms, ground_truths = sinograms.to(device) / phys_scale, ground_truths.to(device) / phys_scale
            optimizer.zero_grad()
            
            reconstructions = model(sinograms)
            
            # Loss Calculation (Data Fidelity + Image Alignment)
            loss = F.mse_loss(reconstructions, ground_truths) + 1e-4 * F.mse_loss(model.physics(reconstructions), sinograms)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            psnr_val, ssim_val = compute_metrics(reconstructions, ground_truths)
            epoch_loss += loss.item()
            epoch_psnr += psnr_val
            epoch_ssim += ssim_val
            steps += 1
            
            if batch_idx % 50 == 0:
                print(f"Ep [{epoch+1}/{epochs}] Stp [{batch_idx}] | Loss: {loss.item():.4f} | PSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f}")
                
            if device == 'cuda' and torch.cuda.get_device_properties(device).total_memory / (1024**3) < 5.0 and steps >= 3:
                print("Local Prototyping Complete.")
                break
                
        scheduler.step()
        epoch_time = time.time() - start_time
        print(f"\n--- {model_name} Epoch {epoch+1} Summary ---")
        print(f"Time: {epoch_time:.1f}s | Avg PSNR: {epoch_psnr/steps:.2f}dB | Avg SSIM: {epoch_ssim/steps:.4f}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define Kaggle / Local Paths
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    MAYO_PATH = "dummy_mayo.h5" # Replace with actual Kaggle path when available
    
    # Determine execution mode based on environment
    if os.path.exists(LODOPAB_PATH):
        print("Kaggle environment detected. Commencing SOTA Benchmarking on LoDoPaB-CT...")
        # 1. Train the Baseline PAUM (Standard CNN)
        train_and_evaluate('PAUM', 'lodopab', LODOPAB_PATH, epochs=5, batch_size=2, device=device)
        
        # 2. Train the Baseline JotlasNet (Transformer)
        train_and_evaluate('JotlasNet', 'lodopab', LODOPAB_PATH, epochs=5, batch_size=2, device=device)
        
        # 3. Train the Proposed PI-KINN (ODE Stateful Network)
        train_and_evaluate('PI_KINN', 'lodopab', LODOPAB_PATH, epochs=25, batch_size=2, device=device)
    else:
        print("Local environment detected. Running pipeline verification...")
        train_and_evaluate('PI_KINN', 'lodopab', 'dummy_path.tfrecord', epochs=1, batch_size=1, device=device)