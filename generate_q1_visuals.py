import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.metrics import structural_similarity as ssim

from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
from physics import RadonPhysics

def create_circular_mask(h, w, device):
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    mask = (X - center[0])**2 + (Y - center[1])**2 <= radius**2
    return torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

def process_for_display(pred_mu, gt_mu, mask):
    """ 
    Safe Mean-Shift Alignment. 
    Aligns the DC bias without exploding the variance.
    """
    p = pred_mu * mask
    g = gt_mu * mask
    
    # 1. Simple Mean-Shift to fix PyTorch FFT DC Bias
    p_mean = p[mask > 0].mean()
    g_mean = g[mask > 0].mean()
    p_shifted = p - p_mean + g_mean
    
    # 2. Normalize to [0, 1] using Ground Truth bounds for fair visual comparison
    g_min = g[mask > 0].min()
    g_max = g[mask > 0].max()
    
    p_norm = torch.clamp((p_shifted - g_min) / (g_max - g_min + 1e-8), 0, 1) * mask
    g_norm = torch.clamp((g - g_min) / (g_max - g_min + 1e-8), 0, 1) * mask
    
    p_np = p_norm.detach().cpu().squeeze().numpy()
    g_np = g_norm.detach().cpu().squeeze().numpy()
    m_np = mask.detach().cpu().squeeze().numpy()
    
    # 3. Calculate Metrics on the normalized visual space
    mse = np.sum(((p_np - g_np)**2) * m_np) / np.sum(m_np)
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))
    ssim_val = ssim(p_np, g_np, data_range=1.0)
    
    # 4. Error Map
    error_map = (p_np - g_np) * m_np
    
    return p_np, g_np, error_map, psnr, ssim_val

def generate_visuals(data_path, device='cuda'):
    print("Generating Q1 Publication Visuals (Safe Mean-Shift)...")
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    fov_mask = create_circular_mask(img_size, img_size, device)
    
    models = {
        "PAUM": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "PI-KINN (Ours)": PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    for name, model in models.items():
        ckpt_name = name.split(" ")[0]
        ckpt_path = f"{ckpt_name}_checkpoint_ep5.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
            model.eval()

    # Grab a deeper slice (Slice 15) for better lung anatomy
    for i, (sino, gt) in enumerate(dataloader):
        if i == 15: 
            sinogram = sino.to(device) / phys_scale
            ground_truth = gt.to(device) / phys_scale
            break

    results = {}
    with torch.no_grad():
        _, gt_img, _, _, _ = process_for_display(ground_truth * phys_scale, ground_truth * phys_scale, fov_mask)
        
        fbp_pred = physics.adjoint(sinogram)
        p_img, _, err, psnr, ssim_v = process_for_display(fbp_pred * phys_scale, ground_truth * phys_scale, fov_mask)
        results["FBP"] = {"img": p_img, "err": err, "psnr": psnr, "ssim": ssim_v}
        
        for name, model in models.items():
            pred = model(sinogram)
            p_img, _, err, psnr, ssim_v = process_for_display(pred * phys_scale, ground_truth * phys_scale, fov_mask)
            results[name] = {"img": p_img, "err": err, "psnr": psnr, "ssim": ssim_v}

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    titles = ["Ground Truth", "FBP", "PAUM", "JotlasNet", "PI-KINN (Ours)"]
    images = [gt_img, results["FBP"]["img"], results["PAUM"]["img"], results["JotlasNet"]["img"], results["PI-KINN (Ours)"]["img"]]
    errors = [np.zeros_like(gt_img), results["FBP"]["err"], results["PAUM"]["err"], results["JotlasNet"]["err"], results["PI-KINN (Ours)"]["err"]]
    
    # ROI Coordinates
    x1, y1, box_size = 150, 150, 60 
    
    for col in range(5):
        # Row 1: Full Image (Using 'bone' colormap for clinical look)
        ax = axes[0, col]
        ax.imshow(images[col], cmap='bone', vmin=0, vmax=1)
        ax.set_title(titles[col], fontsize=16, fontweight='bold')
        ax.axis('off')
        
        rect = patches.Rectangle((x1, y1), box_size, box_size, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        if col > 0:
            metrics_text = f"PSNR: {results[titles[col]]['psnr']:.2f}\nSSIM: {results[titles[col]]['ssim']:.4f}"
            ax.text(10, 340, metrics_text, color='yellow', fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

        # Row 2: Zoomed ROI
        ax_roi = axes[1, col]
        roi_img = images[col][y1:y1+box_size, x1:x1+box_size]
        ax_roi.imshow(roi_img, cmap='bone', vmin=0, vmax=1)
        ax_roi.axis('off')
        for spine in ax_roi.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)

        # Row 3: Error Map
        ax_err = axes[2, col]
        if col == 0:
            ax_err.imshow(np.zeros_like(gt_img), cmap='gray')
            ax_err.set_title("Error Map", fontsize=14)
        else:
            # Tightly bounded error map to show true structural differences
            im_err = ax_err.imshow(errors[col], cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        ax_err.axis('off')

    plt.tight_layout()
    plt.savefig("q1_visual_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    print("SUCCESS: Saved 'q1_visual_comparison.png'.")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        generate_visuals(LODOPAB_PATH, device=device)