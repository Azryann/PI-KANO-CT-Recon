import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.metrics import structural_similarity as ssim

# Import our custom modules
from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
from physics import RadonPhysics

# --- Q1 Math Helpers ---
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

def process_for_display(pred_mu, gt_mu, mask):
    """ Aligns, converts to HU, windows, and returns 2D numpy arrays for plotting. """
    pred_aligned = align_to_clinical_domain(pred_mu, gt_mu, mask)
    
    pred_hu = mu_to_hu(pred_aligned) * mask
    gt_hu = mu_to_hu(gt_mu) * mask
    
    pred_norm = apply_clinical_window(pred_hu) * mask
    gt_norm = apply_clinical_window(gt_hu) * mask
    
    p_np = pred_norm.detach().cpu().squeeze().numpy()
    g_np = gt_norm.detach().cpu().squeeze().numpy()
    m_np = mask.detach().cpu().squeeze().numpy()
    
    mse = np.sum(((p_np - g_np)**2) * m_np) / np.sum(m_np)
    psnr = 10 * np.log10(1.0**2 / (mse + 1e-8))
    ssim_val = ssim(p_np, g_np, data_range=1.0)
    
    # Calculate Error Map in HU space (for clinically meaningful error)
    error_map = (pred_hu.detach().cpu().squeeze().numpy() - gt_hu.detach().cpu().squeeze().numpy()) * m_np
    
    return p_np, g_np, error_map, psnr, ssim_val

def generate_visuals(data_path, device='cuda'):
    print("Generating Q1 Publication Visuals...")
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    fov_mask = create_circular_mask(img_size, img_size, device)
    
    models = {
        "PAUM": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "PI_KINN (Ours)": PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    # Load Weights
    for name, model in models.items():
        ckpt_name = name.split(" ")[0]
        ckpt_path = f"{ckpt_name}_checkpoint_ep5.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
            model.eval()
        else:
            print(f"[ERROR] Missing {ckpt_path}")
            return

    # Grab the 5th slice (usually contains good lung/heart structure)
    for i, (sino, gt) in enumerate(dataloader):
        if i == 4: 
            sinogram = sino.to(device) / phys_scale
            ground_truth = gt.to(device) / phys_scale
            break

    results = {}
    
    with torch.no_grad():
        # 1. Ground Truth
        _, gt_img, _, _, _ = process_for_display(ground_truth * phys_scale, ground_truth * phys_scale, fov_mask)
        
        # 2. FBP
        fbp_pred = physics.adjoint(sinogram)
        p_img, _, err, psnr, ssim_v = process_for_display(fbp_pred * phys_scale, ground_truth * phys_scale, fov_mask)
        results["FBP"] = {"img": p_img, "err": err, "psnr": psnr, "ssim": ssim_v}
        
        # 3. Models
        for name, model in models.items():
            pred = model(sinogram)
            p_img, _, err, psnr, ssim_v = process_for_display(pred * phys_scale, ground_truth * phys_scale, fov_mask)
            results[name] = {"img": p_img, "err": err, "psnr": psnr, "ssim": ssim_v}

    # ==========================================
    # PLOTTING (Q1 Journal Standard)
    # ==========================================
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    titles = ["Ground Truth", "FBP", "PAUM", "JotlasNet", "PI_KINN (Ours)"]
    images = [gt_img, results["FBP"]["img"], results["PAUM"]["img"], results["JotlasNet"]["img"], results["PI_KINN (Ours)"]["img"]]
    errors = [np.zeros_like(gt_img), results["FBP"]["err"], results["PAUM"]["err"], results["JotlasNet"]["err"], results["PI_KINN (Ours)"]["err"]]
    
    # ROI Coordinates (Adjust these if the slice doesn't capture a good feature)
    x1, y1, box_size = 180, 120, 60 
    
    for col in range(5):
        # Row 1: Full Image
        ax = axes[0, col]
        ax.imshow(images[col], cmap='gray', vmin=0, vmax=1)
        ax.set_title(titles[col], fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add Red Bounding Box
        rect = patches.Rectangle((x1, y1), box_size, box_size, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add Metrics Text
        if col > 0:
            metrics_text = f"PSNR: {results[titles[col]]['psnr']:.2f}\nSSIM: {results[titles[col]]['ssim']:.4f}"
            ax.text(10, 340, metrics_text, color='yellow', fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

        # Row 2: Zoomed ROI
        ax_roi = axes[1, col]
        roi_img = images[col][y1:y1+box_size, x1:x1+box_size]
        ax_roi.imshow(roi_img, cmap='gray', vmin=0, vmax=1)
        ax_roi.axis('off')
        # Red border for ROI
        for spine in ax_roi.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)

        # Row 3: Error Map
        ax_err = axes[2, col]
        if col == 0:
            ax_err.imshow(np.zeros_like(gt_img), cmap='gray')
            ax_err.set_title("Error Map", fontsize=14)
        else:
            # RdBu colormap highlights positive errors in Red, negative in Blue
            im_err = ax_err.imshow(errors[col], cmap='RdBu_r', vmin=-200, vmax=200)
        ax_err.axis('off')

    plt.tight_layout()
    plt.savefig("q1_visual_comparison.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'q1_visual_comparison.png'. Download this from Kaggle!")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        generate_visuals(LODOPAB_PATH, device=device)