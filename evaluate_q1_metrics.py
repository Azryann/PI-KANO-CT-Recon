import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Import our custom modules
from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
from physics import RadonPhysics

def create_circular_mask(h, w):
    """ Standard CT Field-of-View (FOV) Mask to ignore corner artifacts. """
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return torch.tensor(mask, dtype=torch.float32)

def calibrate_baseline(pred, gt, mask):
    """ Only used for the untrained FBP baseline to find the global scale. """
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
    # Strict Q1 clinical lung window: [-1000, 400] HU
    clipped = torch.clamp(hu_tensor, min=-1000.0, max=400.0)
    return (clipped + 1000.0) / 1400.0

def compute_q1_metrics(pred_mu, gt_mu, mask, is_baseline=False):
    # 1. Only calibrate FBP. Trained models already learned the scale!
    if is_baseline:
        pred_mu = calibrate_baseline(pred_mu, gt_mu, mask)
    
    # 2. Convert to HU
    pred_hu = mu_to_hu(pred_mu)
    gt_hu = mu_to_hu(gt_mu)
    
    # 3. Apply FOV Mask to HU tensors
    pred_hu = pred_hu * mask
    gt_hu = gt_hu * mask
    
    # Calculate RMSE inside the mask
    rmse_hu = torch.sqrt(torch.sum((pred_hu - gt_hu)**2) / torch.sum(mask)).item()
    
    # 4. Apply Clinical Window [-1000, 400] -> [0, 1]
    pred_norm = apply_clinical_window(pred_hu) * mask
    gt_norm = apply_clinical_window(gt_hu) * mask
    
    p_np = pred_norm.detach().cpu().squeeze().numpy()
    g_np = gt_norm.detach().cpu().squeeze().numpy()
    m_np = mask.detach().cpu().squeeze().numpy()
    
    # 5. Calculate Metrics strictly inside the mask
    mse_norm = np.sum(((p_np - g_np)**2) * m_np) / np.sum(m_np)
    psnr = 10 * np.log10(1.0**2 / (mse_norm + 1e-8))
    
    # SSIM is computed on the whole image (mask sets background to 0, which is standard)
    ssim_val = ssim(p_np, g_np, data_range=1.0)
    
    return psnr, ssim_val, rmse_hu

def evaluate_all_models(data_path, device='cuda', num_test_samples=100):
    print(f"\n{'='*65}\nStarting Q1 Clinical Evaluation (Masked HU Window)\n{'='*65}")
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    fov_mask = create_circular_mask(img_size, img_size).to(device)
    
    models = {
        "PAUM": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "PI_KINN (Ours)": PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    for name, model in models.items():
        ckpt_name = name.split(" ")[0]
        ckpt_path = f"{ckpt_name}_checkpoint_ep5.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
            model.eval()
        else:
            print(f"[WARNING] {ckpt_path} not found!")

    results = {
        "FBP (Baseline)": {"psnr": [], "ssim": [], "rmse": []},
        "PAUM": {"psnr": [], "ssim": [], "rmse": []},
        "JotlasNet": {"psnr": [], "ssim": [], "rmse": []},
        "PI_KINN (Ours)": {"psnr": [], "ssim": [], "rmse": []}
    }
    
    with torch.no_grad():
        for i, (sinogram, gt) in enumerate(dataloader):
            if i >= num_test_samples: break
                
            sinogram = sinogram.to(device) / phys_scale
            gt = gt.to(device) / phys_scale
            
            # FBP (Needs calibration)
            fbp_pred = physics.adjoint(sinogram)
            p, s, r = compute_q1_metrics(fbp_pred * phys_scale, gt * phys_scale, fov_mask, is_baseline=True)
            results["FBP (Baseline)"]["psnr"].append(p)
            results["FBP (Baseline)"]["ssim"].append(s)
            results["FBP (Baseline)"]["rmse"].append(r)
            
            # Deep Learning Models (NO calibration, they learned the scale!)
            for name, model in models.items():
                pred = model(sinogram)
                p, s, r = compute_q1_metrics(pred * phys_scale, gt * phys_scale, fov_mask, is_baseline=False)
                results[name]["psnr"].append(p)
                results[name]["ssim"].append(s)
                results[name]["rmse"].append(r)

    print(f"\n{'='*65}")
    print(f"{'Method':<20} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'RMSE (HU) ↓':<12}")
    print(f"{'-'*65}")
    for name, metrics in results.items():
        avg_psnr = np.mean(metrics['psnr'])
        avg_ssim = np.mean(metrics['ssim'])
        avg_rmse = np.mean(metrics['rmse'])
        if "Ours" in name:
            print(f"\033[1m{name:<20} | {avg_psnr:<12.2f} | {avg_ssim:<10.4f} | {avg_rmse:<12.1f}\033[0m")
        else:
            print(f"{name:<20} | {avg_psnr:<12.2f} | {avg_ssim:<10.4f} | {avg_rmse:<12.1f}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        evaluate_all_models(LODOPAB_PATH, device=device, num_test_samples=100)