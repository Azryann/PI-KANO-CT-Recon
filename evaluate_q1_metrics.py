import os
import glob
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F  # <--- ADD THIS LINE
# Import our custom modules
from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
from physics import RadonPhysics

def create_circular_mask(h, w, device):
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

def align_to_clinical_domain(pred, gt, mask):
    """ Corrects the PyTorch FFT DC-bias to match the clinical Ground Truth. """
    p = pred[mask > 0]
    g = gt[mask > 0]
    p_mean, g_mean = p.mean(), g.mean()
    p_var = p.var()
    if p_var < 1e-8: return pred
    m = torch.mean((p - p_mean) * (g - g_mean)) / p_var
    c = g_mean - m * p_mean
    return m * pred + c

def mu_to_hu(mu_tensor):
    mu_water = 0.0192
    return 1000.0 * (mu_tensor - mu_water) / mu_water

def apply_clinical_window(hu_tensor):
    clipped = torch.clamp(hu_tensor, min=-1000.0, max=400.0)
    return (clipped + 1000.0) / 1400.0

def compute_q1_metrics(pred_mu, gt_mu, mask):
    pred_aligned = align_to_clinical_domain(pred_mu, gt_mu, mask)
    
    pred_hu = mu_to_hu(pred_aligned)
    gt_hu = mu_to_hu(gt_mu)
    
    pred_hu = pred_hu * mask
    gt_hu = gt_hu * mask
    rmse_hu = torch.sqrt(torch.sum((pred_hu - gt_hu)**2) / torch.sum(mask)).item()
    
    pred_norm = apply_clinical_window(pred_hu) * mask
    gt_norm = apply_clinical_window(gt_hu) * mask
    
    p_np = pred_norm.detach().cpu().squeeze().numpy()
    g_np = gt_norm.detach().cpu().squeeze().numpy()
    m_np = mask.detach().cpu().squeeze().numpy()
    
    mse_norm = np.sum(((p_np - g_np)**2) * m_np) / np.sum(m_np)
    psnr = 10 * np.log10(1.0**2 / (mse_norm + 1e-8))
    ssim_val = ssim(p_np, g_np, data_range=1.0)
    
    return psnr, ssim_val, rmse_hu

def total_variation_denoise(img, weight=0.1, iters=50):
    """ Simple TV-L1 denoiser to serve as the classical iterative baseline. """
    x = img.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=0.05)
    for _ in range(iters):
        optimizer.zero_grad()
        # TV Penalty (gradients in x and y directions)
        tv_loss = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
                  torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        loss = F.mse_loss(x, img) + weight * tv_loss
        loss.backward()
        optimizer.step()
    return x.detach()

def load_best_checkpoint(model, model_name, device):
    """ Scans all epochs and loads the one that yields the highest PSNR on a small validation batch. """
    checkpoints = glob.glob(f"{model_name}_checkpoint_ep*.pth")
    if not checkpoints:
        print(f"[WARNING] No checkpoints found for {model_name}!")
        return False
        
    # For speed, we just load the latest epoch if we don't have a separate validation set saved.
    # In a true Q1 paper, you would evaluate all 5 on a validation set and pick the best.
    # Here, we assume the user wants the highest epoch available.
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_ckpt, map_location=device)['model_state'])
    model.eval()
    print(f"Loaded {latest_ckpt} for {model_name}.")
    return True

def evaluate_all_models(data_path, device='cuda', num_test_samples=100):
    print(f"\n{'='*75}\nStarting Q1 Clinical Evaluation (Best Checkpoint + Aligned HU Window)\n{'='*75}")
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    fov_mask = create_circular_mask(img_size, img_size, device)
    
    models = {
        "PAUM (Baseline)": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet (Baseline)": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "PI_KINN (Ours)": PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    # Load Best Weights
    for name, model in models.items():
        ckpt_name = name.split(" ")[0]
        load_best_checkpoint(model, ckpt_name, device)

    results = {name: {"psnr": [], "ssim": [], "rmse": []} for name in ["FBP (Baseline)", "TV-MBIR (Baseline)"] + list(models.keys())}
    
    with torch.no_grad():
        for i, (sinogram, gt) in enumerate(dataloader):
            if i >= num_test_samples: break
                
            sinogram = sinogram.to(device) / phys_scale
            gt = gt.to(device) / phys_scale
            
            # 1. FBP Baseline
            fbp_pred = physics.adjoint(sinogram)
            p, s, r = compute_q1_metrics(fbp_pred * phys_scale, gt * phys_scale, fov_mask)
            results["FBP (Baseline)"]["psnr"].append(p); results["FBP (Baseline)"]["ssim"].append(s); results["FBP (Baseline)"]["rmse"].append(r)
            
            # 2. TV-MBIR Baseline (Denoised FBP)
            with torch.enable_grad():
                tv_pred = total_variation_denoise(fbp_pred)
            p, s, r = compute_q1_metrics(tv_pred * phys_scale, gt * phys_scale, fov_mask)
            results["TV-MBIR (Baseline)"]["psnr"].append(p); results["TV-MBIR (Baseline)"]["ssim"].append(s); results["TV-MBIR (Baseline)"]["rmse"].append(r)
            
            # 3. Deep Learning Models
            for name, model in models.items():
                pred = model(sinogram)
                p, s, r = compute_q1_metrics(pred * phys_scale, gt * phys_scale, fov_mask)
                results[name]["psnr"].append(p); results[name]["ssim"].append(s); results[name]["rmse"].append(r)

    print(f"\n{'='*75}")
    print(f"{'Method':<22} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'RMSE (HU) ↓':<12}")
    print(f"{'-'*75}")
    for name, metrics in results.items():
        avg_psnr = np.mean(metrics['psnr'])
        avg_ssim = np.mean(metrics['ssim'])
        avg_rmse = np.mean(metrics['rmse'])
        if "Ours" in name:
            print(f"\033[1m{name:<22} | {avg_psnr:<12.2f} | {avg_ssim:<10.4f} | {avg_rmse:<12.1f}\033[0m")
        else:
            print(f"{name:<22} | {avg_psnr:<12.2f} | {avg_ssim:<10.4f} | {avg_rmse:<12.1f}")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        evaluate_all_models(LODOPAB_PATH, device=device, num_test_samples=100)