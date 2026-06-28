import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

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

def mu_to_hu(mu_tensor):
    mu_water = 0.0192
    return 1000.0 * (mu_tensor - mu_water) / mu_water

def apply_clinical_window(hu_tensor):
    clipped = torch.clamp(hu_tensor, min=-1000.0, max=400.0)
    return (clipped + 1000.0) / 1400.0

def compute_dual_metrics(pred_mu, gt_mu, mask):
    """ Computes metrics in BOTH the Optimization Domain (mu) and Clinical Domain (HU). """
    
    # ==========================================
    # 1. OPTIMIZATION DOMAIN (mu-space)
    # ==========================================
    # Mask the raw tensors
    p_mu_m = pred_mu * mask
    g_mu_m = gt_mu * mask
    
    p_mu_np = p_mu_m.detach().cpu().squeeze().numpy()
    g_mu_np = g_mu_m.detach().cpu().squeeze().numpy()
    m_np = mask.detach().cpu().squeeze().numpy()
    
    # Calculate data range dynamically based on Ground Truth mu
    mu_range = g_mu_np.max() - g_mu_np.min() + 1e-8
    
    mse_mu = np.sum(((p_mu_np - g_mu_np)**2) * m_np) / np.sum(m_np)
    psnr_mu = 10 * np.log10((mu_range**2) / (mse_mu + 1e-8))
    ssim_mu = ssim(p_mu_np, g_mu_np, data_range=mu_range)
    
    # ==========================================
    # 2. CLINICAL DOMAIN (HU-space Windowed)
    # ==========================================
    pred_hu = mu_to_hu(pred_mu)
    gt_hu = mu_to_hu(gt_mu)
    
    rmse_hu = torch.sqrt(torch.sum(((pred_hu - gt_hu) * mask)**2) / torch.sum(mask)).item()
    
    pred_norm = apply_clinical_window(pred_hu) * mask
    gt_norm = apply_clinical_window(gt_hu) * mask
    
    p_hu_np = pred_norm.detach().cpu().squeeze().numpy()
    g_hu_np = gt_norm.detach().cpu().squeeze().numpy()
    
    mse_hu_norm = np.sum(((p_hu_np - g_hu_np)**2) * m_np) / np.sum(m_np)
    psnr_hu = 10 * np.log10(1.0**2 / (mse_hu_norm + 1e-8))
    ssim_hu = ssim(p_hu_np, g_hu_np, data_range=1.0)
    
    return psnr_mu, ssim_mu, psnr_hu, ssim_hu, rmse_hu

def evaluate_all_models(data_path, device='cuda', num_test_samples=100):
    print(f"\n{'='*75}\nStarting Dual-Domain Q1 Evaluation\n{'='*75}")
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    fov_mask = create_circular_mask(img_size, img_size, device)
    
    models = {
        "PAUM (Baseline)": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet (Baseline)": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "PI-KINN (Ours)": PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    for name, model in models.items():
        ckpt_name = name.split(" ")[0]
        ckpt_path = f"{ckpt_name}_checkpoint_ep5.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
            model.eval()

    results = {name: {"p_mu": [], "s_mu": [], "p_hu": [], "s_hu": [], "r_hu": []} for name in ["FBP (Baseline)"] + list(models.keys())}
    
    with torch.no_grad():
        for i, (sinogram, gt) in enumerate(dataloader):
            if i >= num_test_samples: break
                
            sinogram = sinogram.to(device) / phys_scale
            gt = gt.to(device) / phys_scale
            
            # FBP
            fbp_pred = physics.adjoint(sinogram)
            p_m, s_m, p_h, s_h, r_h = compute_dual_metrics(fbp_pred * phys_scale, gt * phys_scale, fov_mask)
            results["FBP (Baseline)"]["p_mu"].append(p_m); results["FBP (Baseline)"]["s_mu"].append(s_m)
            results["FBP (Baseline)"]["p_hu"].append(p_h); results["FBP (Baseline)"]["s_hu"].append(s_h); results["FBP (Baseline)"]["r_hu"].append(r_h)
            
            # Deep Learning Models
            for name, model in models.items():
                pred = model(sinogram)
                p_m, s_m, p_h, s_h, r_h = compute_dual_metrics(pred * phys_scale, gt * phys_scale, fov_mask)
                results[name]["p_mu"].append(p_m); results[name]["s_mu"].append(s_m)
                results[name]["p_hu"].append(p_h); results[name]["s_hu"].append(s_h); results[name]["r_hu"].append(r_h)

    print(f"\n{'='*75}")
    print(f"TABLE 1: RECONSTRUCTION FIDELITY (Optimization Domain: μ-space)")
    print(f"{'-'*75}")
    print(f"{'Method':<22} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10}")
    print(f"{'-'*75}")
    for name, metrics in results.items():
        p, s = np.mean(metrics['p_mu']), np.mean(metrics['s_mu'])
        if "Ours" in name: print(f"\033[1m{name:<22} | {p:<12.2f} | {s:<10.4f}\033[0m")
        else: print(f"{name:<22} | {p:<12.2f} | {s:<10.4f}")

    print(f"\n{'='*75}")
    print(f"TABLE 2: CLINICAL FIDELITY (Clinical Domain: Windowed HU-space)")
    print(f"{'-'*75}")
    print(f"{'Method':<22} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'RMSE (HU) ↓':<12}")
    print(f"{'-'*75}")
    for name, metrics in results.items():
        p, s, r = np.mean(metrics['p_hu']), np.mean(metrics['s_hu']), np.mean(metrics['r_hu'])
        if "Ours" in name: print(f"\033[1m{name:<22} | {p:<12.2f} | {s:<10.4f} | {r:<12.1f}\033[0m")
        else: print(f"{name:<22} | {p:<12.2f} | {s:<10.4f} | {r:<12.1f}")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        evaluate_all_models(LODOPAB_PATH, device=device, num_test_samples=100)