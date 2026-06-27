import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Import our custom modules
from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
from physics import RadonPhysics

def calibrate_tensor(pred, gt):
    """
    Global Affine Calibration (Least Squares Fit).
    Corrects the global physics scaling mismatch before HU conversion,
    revealing the true structural fidelity of the reconstruction.
    """
    p = pred.flatten()
    g = gt.flatten()
    
    p_mean = p.mean()
    g_mean = g.mean()
    p_var = p.var()
    
    if p_var < 1e-8:
        return pred - p_mean + g_mean
        
    # y = mx + c
    m = torch.mean((p - p_mean) * (g - g_mean)) / p_var
    c = g_mean - m * p_mean
    
    return m * pred + c

def mu_to_hu(mu_tensor):
    mu_water = 0.0192
    return 1000.0 * (mu_tensor - mu_water) / mu_water

def apply_clinical_window(hu_tensor):
    # Strict Q1 clinical lung window: [-1000, 400] HU
    clipped = torch.clamp(hu_tensor, min=-1000.0, max=400.0)
    # Normalize to [0, 1]
    return (clipped + 1000.0) / 1400.0

def compute_q1_metrics(pred_mu, gt_mu):
    # 1. Calibrate the physics scale
    pred_calibrated = calibrate_tensor(pred_mu, gt_mu)
    
    # 2. Convert to HU
    pred_hu = mu_to_hu(pred_calibrated)
    gt_hu = mu_to_hu(gt_mu)
    
    rmse_hu = torch.sqrt(torch.mean((pred_hu - gt_hu) ** 2)).item()
    
    # 3. Apply Clinical Window
    pred_norm = apply_clinical_window(pred_hu).detach().cpu().squeeze().numpy()
    gt_norm = apply_clinical_window(gt_hu).detach().cpu().squeeze().numpy()
    
    # 4. Calculate Metrics
    mse_norm = np.mean((pred_norm - gt_norm) ** 2)
    psnr = 10 * np.log10(1.0**2 / (mse_norm + 1e-8))
    ssim_val = ssim(pred_norm, gt_norm, data_range=1.0)
    
    return psnr, ssim_val, rmse_hu

def evaluate_all_models(data_path, device='cuda', num_test_samples=100):
    print(f"\n{'='*65}\nStarting Q1 Clinical Evaluation (Calibrated HU Window)\n{'='*65}")
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    
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
        "PI-KINN (Ours)": {"psnr": [], "ssim": [], "rmse": []}
    }
    
    with torch.no_grad():
        for i, (sinogram, gt) in enumerate(dataloader):
            if i >= num_test_samples: break
                
            sinogram = sinogram.to(device) / phys_scale
            gt = gt.to(device) / phys_scale
            
            # FBP
            fbp_pred = physics.adjoint(sinogram)
            p, s, r = compute_q1_metrics(fbp_pred * phys_scale, gt * phys_scale)
            results["FBP (Baseline)"]["psnr"].append(p)
            results["FBP (Baseline)"]["ssim"].append(s)
            results["FBP (Baseline)"]["rmse"].append(r)
            
            # Deep Learning Models
            for name, model in models.items():
                pred = model(sinogram)
                p, s, r = compute_q1_metrics(pred * phys_scale, gt * phys_scale)
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