import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time

# Import our custom modules
from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
from physics import RadonPhysics

def mu_to_hu(mu_tensor):
    """
    Converts linear attenuation coefficients (mu) to Hounsfield Units (HU).
    Standard water attenuation for CT is approx 0.0192 mm^-1.
    HU = 1000 * (mu - mu_water) / mu_water
    """
    mu_water = 0.0192
    hu_tensor = 1000.0 * (mu_tensor - mu_water) / mu_water
    return hu_tensor

def apply_clinical_window(hu_tensor):
    """
    Applies the strict Q1 clinical lung window: [-1000, 400] HU.
    Normalizes the output to [0, 1] for metric computation.
    """
    # 1. Clip to [-1000, 400]
    clipped = torch.clamp(hu_tensor, min=-1000.0, max=400.0)
    # 2. Normalize to [0, 1] (Range = 1400)
    normalized = (clipped + 1000.0) / 1400.0
    return normalized

def compute_q1_metrics(pred_mu, gt_mu):
    """
    Computes PSNR, SSIM, and RMSE strictly on the windowed [0, 1] HU scale.
    """
    # Convert to HU
    pred_hu = mu_to_hu(pred_mu)
    gt_hu = mu_to_hu(gt_mu)
    
    # Calculate RMSE in raw HU space (Standard clinical metric)
    rmse_hu = torch.sqrt(torch.mean((pred_hu - gt_hu) ** 2)).item()
    
    # Apply Clinical Window [-1000, 400] -> [0, 1]
    pred_norm = apply_clinical_window(pred_hu).detach().cpu().squeeze().numpy()
    gt_norm = apply_clinical_window(gt_hu).detach().cpu().squeeze().numpy()
    
    # Calculate PSNR on [0, 1] scale
    mse_norm = np.mean((pred_norm - gt_norm) ** 2)
    psnr = 10 * np.log10(1.0**2 / (mse_norm + 1e-8))
    
    # Calculate SSIM on [0, 1] scale
    ssim_val = ssim(pred_norm, gt_norm, data_range=1.0)
    
    return psnr, ssim_val, rmse_hu

def evaluate_all_models(data_path, device='cuda', num_test_samples=100):
    print(f"\n{'='*60}\nStarting Q1 Clinical Evaluation (HU Windowed)\n{'='*60}")
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    
    # Initialize Models
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    
    models = {
        "PAUM": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "PI-KINN (Ours)": PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    # Load Weights
    for name, model in models.items():
        ckpt_name = name.split(" ")[0] # Handles "PI-KINN (Ours)" -> "PI-KINN"
        ckpt_path = f"{ckpt_name}_checkpoint_ep5.pth"
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            print(f"Loaded {name} weights successfully.")
        else:
            print(f"[WARNING] {ckpt_path} not found. Model will use random weights!")

    # Metric Dictionaries
    results = {
        "FBP (Baseline)": {"psnr": [], "ssim": [], "rmse": []},
        "PAUM": {"psnr": [], "ssim": [], "rmse": []},
        "JotlasNet": {"psnr": [], "ssim": [], "rmse": []},
        "PI-KINN (Ours)": {"psnr": [], "ssim": [], "rmse": []}
    }
    
    print(f"\nEvaluating on {num_test_samples} test slices...")
    with torch.no_grad():
        for i, (sinogram, gt) in enumerate(dataloader):
            if i >= num_test_samples:
                break
                
            sinogram = sinogram.to(device) / phys_scale
            gt = gt.to(device) / phys_scale
            
            # 1. FBP Baseline (Direct Adjoint of DFST)
            fbp_pred = physics.adjoint(sinogram)
            p, s, r = compute_q1_metrics(fbp_pred * phys_scale, gt * phys_scale)
            results["FBP (Baseline)"]["psnr"].append(p)
            results["FBP (Baseline)"]["ssim"].append(s)
            results["FBP (Baseline)"]["rmse"].append(r)
            
            # 2. Deep Learning Models
            for name, model in models.items():
                pred = model(sinogram)
                p, s, r = compute_q1_metrics(pred * phys_scale, gt * phys_scale)
                results[name]["psnr"].append(p)
                results[name]["ssim"].append(s)
                results[name]["rmse"].append(r)
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_test_samples} slices...")

    # Print Final Q1 Table
    print(f"\n{'='*65}")
    print(f"{'Method':<20} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'RMSE (HU) ↓':<12}")
    print(f"{'-'*65}")
    for name, metrics in results.items():
        avg_psnr = np.mean(metrics['psnr'])
        avg_ssim = np.mean(metrics['ssim'])
        avg_rmse = np.mean(metrics['rmse'])
        
        # Highlight PI-KINN
        if "Ours" in name:
            print(f"\033[1m{name:<20} | {avg_psnr:<12.2f} | {avg_ssim:<10.4f} | {avg_rmse:<12.1f}\033[0m")
        else:
            print(f"{name:<20} | {avg_psnr:<12.2f} | {avg_ssim:<10.4f} | {avg_rmse:<12.1f}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    
    if os.path.exists(LODOPAB_PATH):
        # Evaluates 100 slices to give a highly accurate, fast statistical mean
        evaluate_all_models(LODOPAB_PATH, device=device, num_test_samples=100)
    else:
        print("Kaggle dataset not found. Run on Kaggle to evaluate.")