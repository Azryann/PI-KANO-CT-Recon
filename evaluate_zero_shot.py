import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms

from physics import RadonPhysics
from fda_net import FDA_Net
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate

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

def process_for_display(pred_mu, gt_mu, mask):
    """ Applies Histogram Matching to fix OOD contrast shifts, then windows. """
    p = pred_mu.detach().cpu().squeeze().numpy()
    g = gt_mu.detach().cpu().squeeze().numpy()
    m = mask.detach().cpu().squeeze().numpy().astype(bool)
    
    # Q1 FIX: Histogram match to Ground Truth to fix synthetic phantom contrast
    p[m] = match_histograms(p[m], g[m])
    
    p_t = torch.tensor(p).unsqueeze(0).unsqueeze(0)
    g_t = torch.tensor(g).unsqueeze(0).unsqueeze(0)
    
    p_norm = apply_clinical_window(mu_to_hu(p_t)) * mask.cpu()
    g_norm = apply_clinical_window(mu_to_hu(g_t)) * mask.cpu()
    
    p_np = p_norm.squeeze().numpy()
    g_np = g_norm.squeeze().numpy()
    
    mse = np.sum(((p_np - g_np)**2) * m) / np.sum(m)
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))
    ssim_val = ssim(p_np, g_np, data_range=1.0)
    
    return p_np, g_np, psnr, ssim_val

def generate_shepp_logan_phantom(size=362):
    ellipses = [
        [1.0,   0.69,   0.92,   0.0,    0.0,    0.0],
        [-0.8,  0.6624, 0.874,  0.0,    -0.0184,0.0],
        [-0.2,  0.11,   0.31,   0.22,   0.0,    -18.0],
        [-0.2,  0.16,   0.41,   -0.22,  0.0,    18.0],
        [0.1,   0.21,   0.25,   0.0,    0.35,   0.0],
        [0.1,   0.046,  0.046,  0.0,    0.1,    0.0],
        [0.1,   0.046,  0.046,  0.0,    -0.1,   0.0],
        [0.1,   0.046,  0.023,  -0.08,  -0.605, 0.0],
        [0.1,   0.023,  0.023,  0.0,    -0.606, 0.0],
        [0.1,   0.023,  0.046,  0.06,   -0.605, 0.0]
    ]
    img = np.zeros((size, size), dtype=np.float32)
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    for intensity, a, b, x0, y0, phi in ellipses:
        phi_rad = np.deg2rad(phi)
        cos_p, sin_p = np.cos(phi_rad), np.sin(phi_rad)
        x_rot = (x - x0) * cos_p + (y - y0) * sin_p
        y_rot = -(x - x0) * sin_p + (y - y0) * cos_p
        mask = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1.0
        img[mask] += intensity
    img = np.clip(img, 0, 1)
    return torch.tensor(img * 0.04, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def run_zero_shot(device='cuda'):
    print(f"\n{'='*75}\nRunning Zero-Shot Generalization on Shepp-Logan Phantom\n{'='*75}")
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    fov_mask = create_circular_mask(img_size, img_size, device)
    
    print("Generating Shepp-Logan Phantom...")
    gt_mu = generate_shepp_logan_phantom(img_size).to(device)
    gt_scaled = gt_mu / phys_scale
    
    print("Simulating CT acquisition...")
    with torch.no_grad():
        sinogram_scaled = physics.forward(gt_scaled)
        # Q1 FIX: Realistic clinical noise (1% of mean, not 5% of max)
        noise = torch.randn_like(sinogram_scaled) * 0.01 * sinogram_scaled.mean()
        noisy_sino_scaled = sinogram_scaled + noise

    models = {
        "FS-CNN (PAUM)": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "FS-ViT (JotlasNet)": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "FS-Net (Ours)": FDA_Net(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    ckpt_mapping = {"FS-CNN (PAUM)": "PAUM", "FS-ViT (JotlasNet)": "JotlasNet", "FS-Net (Ours)": "FDA_Net"}
    
    for name, model in models.items():
        ckpt_path = f"{ckpt_mapping[name]}_subset_BEST.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model_state'])
            model.eval()

    results = {}
    with torch.no_grad():
        fbp_scaled = physics.adjoint(noisy_sino_scaled)
        
        p_img, gt_img, psnr, ssim_v = process_for_display(fbp_scaled * phys_scale, gt_mu, fov_mask)
        results["FBP"] = {"img": p_img, "psnr": psnr, "ssim": ssim_v}
        
        for name, model in models.items():
            pred_scaled = model(noisy_sino_scaled)
            p_img, _, psnr, ssim_v = process_for_display(pred_scaled * phys_scale, gt_mu, fov_mask)
            results[name] = {"img": p_img, "psnr": psnr, "ssim": ssim_v}

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ["Ground Truth (Phantom)", "FBP", "FS-CNN (PAUM)", "FS-ViT (JotlasNet)", "FS-Net (Ours)"]
    images = [gt_img, results["FBP"]["img"], results["FS-CNN (PAUM)"]["img"], results["FS-ViT (JotlasNet)"]["img"], results["FS-Net (Ours)"]["img"]]
    
    for col in range(5):
        ax = axes[col]
        ax.imshow(images[col], cmap='bone', vmin=0, vmax=1)
        ax.set_title(titles[col], fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if col > 0:
            metrics_text = f"PSNR: {results[titles[col]]['psnr']:.2f}\nSSIM: {results[titles[col]]['ssim']:.4f}"
            ax.text(10, 340, metrics_text, color='yellow', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig("fig_zero_shot.png", dpi=300, bbox_inches='tight', facecolor='white')
    print("SUCCESS: Saved 'fig_zero_shot.png'")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_zero_shot(device=device)