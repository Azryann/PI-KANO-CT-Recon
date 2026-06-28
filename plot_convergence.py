import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Import our custom modules
from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
from physics import RadonPhysics

# --- Re-use the exact Q1 Math from our evaluation script ---
def create_circular_mask(h, w, device):
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    mask = (X - center[0])**2 + (Y - center[1])**2 <= radius**2
    return torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

def align_to_clinical_domain(pred, gt, mask):
    p, g = pred[mask > 0], gt[mask > 0]
    p_mean, g_mean = p.mean(), g.mean()
    p_var = p.var()
    if p_var < 1e-8: return pred
    m = torch.mean((p - p_mean) * (g - g_mean)) / p_var
    return m * pred + (g_mean - m * p_mean)

def mu_to_hu(mu_tensor):
    return 1000.0 * (mu_tensor - 0.0192) / 0.0192

def apply_clinical_window(hu_tensor):
    return (torch.clamp(hu_tensor, min=-1000.0, max=400.0) + 1000.0) / 1400.0

def compute_psnr(pred_mu, gt_mu, mask):
    pred_aligned = align_to_clinical_domain(pred_mu, gt_mu, mask)
    pred_hu = mu_to_hu(pred_aligned) * mask
    gt_hu = mu_to_hu(gt_mu) * mask
    
    pred_norm = apply_clinical_window(pred_hu) * mask
    gt_norm = apply_clinical_window(gt_hu) * mask
    
    p_np = pred_norm.detach().cpu().squeeze().numpy()
    g_np = gt_norm.detach().cpu().squeeze().numpy()
    m_np = mask.detach().cpu().squeeze().numpy()
    
    mse_norm = np.sum(((p_np - g_np)**2) * m_np) / np.sum(m_np)
    return 10 * np.log10(1.0**2 / (mse_norm + 1e-8))

def generate_convergence_curves(data_path, device='cuda'):
    print("Generating Q1 Convergence Curves from Checkpoints...")
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    
    # Use a small batch of 20 slices for fast evaluation
    num_eval_slices = 20 
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    
    # Extract 20 slices into memory so we evaluate all models on the EXACT same data
    eval_data = []
    for i, (sino, gt) in enumerate(dataloader):
        if i >= num_test_slices: break
        eval_data.append((sino.to(device) / phys_scale, gt.to(device) / phys_scale))
        
    fov_mask = create_circular_mask(img_size, img_size, device)
    
    models = {
        "PAUM": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "PI-KINN": PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    history = {"PAUM": [], "JotlasNet": [], "PI-KINN": []}
    epochs = [1, 2, 3, 4, 5]
    
    with torch.no_grad():
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            for ep in epochs:
                ckpt_path = f"{name}_checkpoint_ep{ep}.pth"
                if not os.path.exists(ckpt_path):
                    print(f"  [Missing] {ckpt_path}")
                    history[name].append(None)
                    continue
                
                model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
                model.eval()
                
                ep_psnr = 0.0
                for sino, gt in eval_data:
                    pred = model(sino)
                    ep_psnr += compute_psnr(pred * phys_scale, gt * phys_scale, fov_mask)
                
                avg_psnr = ep_psnr / num_eval_slices
                history[name].append(avg_psnr)
                print(f"  Epoch {ep}: {avg_psnr:.2f} dB")

    # --- PLOTTING ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 5))
    
    colors = {"PAUM": "#1f77b4", "JotlasNet": "#ff7f0e", "PI-KINN": "#d62728"}
    markers = {"PAUM": "s", "JotlasNet": "^", "PI-KINN": "o"}
    
    for name in models.keys():
        valid_eps = [e for e, p in zip(epochs, history[name]) if p is not None]
        valid_psnrs = [p for p in history[name] if p is not None]
        if valid_psnrs:
            plt.plot(valid_eps, valid_psnrs, label=name, color=colors[name], 
                     marker=markers[name], linewidth=2, markersize=8)

    plt.title("Validation Convergence (Clinical HU-Windowed PSNR)", fontsize=14, fontweight='bold')
    plt.xlabel("Training Epochs (1 Epoch ≈ 17,900 steps)", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    
    plt.savefig("convergence_curve.png", dpi=300)
    print("\nSUCCESS: Saved 'convergence_curve.png'.")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        generate_convergence_curves(LODOPAB_PATH, device=device)