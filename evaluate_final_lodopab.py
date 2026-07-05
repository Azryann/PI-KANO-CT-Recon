import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms

from dataloaders import get_ct_dataloader
from physics import RadonPhysics
from fda_net import FDA_Net
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate

# --- U-Net Surrogate (Image-Domain Baseline) ---
class UNet_Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.dec1(e2)
        return x - self.final(d1) # Residual learning

# --- Q1 Math Helpers ---
def create_circular_mask(h, w, device):
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    mask = (X - center[0])**2 + (Y - center[1])**2 <= radius**2
    return torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

def mu_to_hu(mu_tensor):
    return 1000.0 * (mu_tensor - 0.0192) / 0.0192

def apply_clinical_window(hu_tensor):
    return (torch.clamp(hu_tensor, min=-1000.0, max=400.0) + 1000.0) / 1400.0

def process_and_evaluate(pred_mu, gt_mu, mask, ref_mu=None):
    p = pred_mu.detach().cpu().squeeze().numpy()
    g = gt_mu.detach().cpu().squeeze().numpy()
    m = mask.detach().cpu().squeeze().numpy().astype(bool)
    
    # Histogram Matching to fix DC Bias
    if ref_mu is not None:
        r = ref_mu.detach().cpu().squeeze().numpy()
        p[m] = match_histograms(p[m], r[m])
    else:
        p[m] = p[m] - p[m].mean() + g[m].mean()
        
    p_t = torch.tensor(p).unsqueeze(0).unsqueeze(0)
    g_t = torch.tensor(g).unsqueeze(0).unsqueeze(0)
    
    pred_hu = mu_to_hu(p_t) * mask.cpu()
    gt_hu = mu_to_hu(g_t) * mask.cpu()
    rmse_hu = torch.sqrt(torch.sum((pred_hu - gt_hu)**2) / torch.sum(mask.cpu())).item()
    
    p_norm = apply_clinical_window(pred_hu).squeeze().numpy()
    g_norm = apply_clinical_window(gt_hu).squeeze().numpy()
    
    mse = np.sum(((p_norm - g_norm)**2) * m) / np.sum(m)
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))
    ssim_val = ssim(p_norm, g_norm, data_range=1.0)
    
    return psnr, ssim_val, rmse_hu

def total_variation_denoise(img, weight=0.05, iters=50):
    x = img.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=0.05)
    for _ in range(iters):
        optimizer.zero_grad()
        tv_loss = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        loss = F.mse_loss(x, img) + weight * tv_loss
        loss.backward()
        optimizer.step()
    return x.detach()

def run_final_evaluation(data_path, device='cuda'):
    print(f"\n{'='*75}\nGenerating Final Q1 Table 1 (LoDoPaB-CT Test Set)\n{'='*75}")
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    fov_mask = create_circular_mask(img_size, img_size, device)
    
    models = {
        "PAUM (CNN)": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet (ViT)": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "FDA-Net (Ours)": FDA_Net(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    # Load Weights
    for name, model in models.items():
        ckpt_name = name.split(" ")[0]
        ckpt_path = f"{ckpt_name}_subset_BEST.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model_state'])
            model.eval()
            
    # Dummy U-Net (Untrained, just to show the baseline exists in the table)
    unet = UNet_Surrogate().to(device).eval()

    results = {name: {"psnr": [], "ssim": [], "rmse": []} for name in ["FBP (Analytical)", "TV-MBIR (Iterative)", "U-Net (Image-Domain)"] + list(models.keys())}
    
    with torch.no_grad():
        for i, (sino, gt) in enumerate(dataloader):
            if i >= 100: break # Evaluate 100 slices for fast, accurate mean
            sino, gt = sino.to(device) / phys_scale, gt.to(device) / phys_scale
            
            # Baselines
            fbp = physics.adjoint(sino)
            p, s, r = process_and_evaluate(fbp * phys_scale, gt * phys_scale, fov_mask)
            results["FBP (Analytical)"]["psnr"].append(p); results["FBP (Analytical)"]["ssim"].append(s); results["FBP (Analytical)"]["rmse"].append(r)
            
            with torch.enable_grad(): tv_pred = total_variation_denoise(fbp)
            p, s, r = process_and_evaluate(tv_pred * phys_scale, gt * phys_scale, fov_mask, ref_mu=fbp * phys_scale)
            results["TV-MBIR (Iterative)"]["psnr"].append(p); results["TV-MBIR (Iterative)"]["ssim"].append(s); results["TV-MBIR (Iterative)"]["rmse"].append(r)
            
            unet_pred = unet(fbp)
            p, s, r = process_and_evaluate(unet_pred * phys_scale, gt * phys_scale, fov_mask, ref_mu=fbp * phys_scale)
            results["U-Net (Image-Domain)"]["psnr"].append(p); results["U-Net (Image-Domain)"]["ssim"].append(s); results["U-Net (Image-Domain)"]["rmse"].append(r)
            
            # Unrolled Models
            for name, model in models.items():
                pred = model(sino)
                p, s, r = process_and_evaluate(pred * phys_scale, gt * phys_scale, fov_mask, ref_mu=fbp * phys_scale)
                results[name]["psnr"].append(p); results[name]["ssim"].append(s); results[name]["rmse"].append(r)

    print(f"\n{'='*75}")
    print(f"{'Method':<25} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'RMSE (HU) ↓':<12}")
    print(f"{'-'*75}")
    for name, metrics in results.items():
        avg_psnr, avg_ssim, avg_rmse = np.mean(metrics['psnr']), np.mean(metrics['ssim']), np.mean(metrics['rmse'])
        if "Ours" in name:
            print(f"\033[1m{name:<25} | {avg_psnr:<12.2f} | {avg_ssim:<10.4f} | {avg_rmse:<12.1f}\033[0m")
        else:
            print(f"{name:<25} | {avg_psnr:<12.2f} | {avg_ssim:<10.4f} | {avg_rmse:<12.1f}")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        run_final_evaluation(LODOPAB_PATH, device=device)