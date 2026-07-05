import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# Assuming these are your custom imports
# from dataloaders import get_ct_dataloader
# from physics import RadonPhysics
# from fda_net import FDA_Net
# from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate

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

def process_for_raps(pred_mu, gt_mu, mask):
    """
    CRITICAL Q1 FIX: Removed histogram matching which masked the baseline blurring artifacts.
    Standardized solely on mean-shifting to isolate true model frequency response.
    """
    p = pred_mu.detach().cpu().squeeze().numpy()
    g = gt_mu.detach().cpu().squeeze().numpy()
    m = mask.detach().cpu().squeeze().numpy().astype(bool)
    
    # Zero-mean shift within the reconstruction FOV mask to match DC bias fairly
    p[m] = p[m] - p[m].mean() + g[m].mean()
        
    p_t = torch.tensor(p).unsqueeze(0).unsqueeze(0)
    g_t = torch.tensor(g).unsqueeze(0).unsqueeze(0)
    
    p_norm = apply_clinical_window(mu_to_hu(p_t)) * mask.cpu()
    g_norm = apply_clinical_window(mu_to_hu(g_t)) * mask.cpu()
    
    return p_norm.squeeze().numpy(), g_norm.squeeze().numpy()

def compute_raps(image_np):
    f = np.fft.fft2(image_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)**2
    
    h, w = magnitude_spectrum.shape
    y, x = np.indices((h, w))
    center = (int(h/2), int(w/2))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    
    # Normalize to DC component for standard relative structural comparison
    radialprofile = radialprofile / radialprofile[0]
    return np.log10(radialprofile + 1e-12)

def generate_full_raps(data_path, device='cuda'):
    print("Generating Q1 RAPS Plot (Clinically Windowed)...")
    
    # Journal-ready publication typography configuration
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11
    })
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    fov_mask = create_circular_mask(img_size, img_size, device)
    
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
        else:
            print(f"[WARNING] {ckpt_path} not found!")

    for i, (sino, gt) in enumerate(dataloader):
        if i == 15:  # Selecting targeted structural scan
            sinogram = sino.to(device) / phys_scale
            ground_truth = gt.to(device) / phys_scale
            break

    with torch.no_grad():
        fbp_pred = physics.adjoint(sinogram)
        _, gt_img = process_for_raps(ground_truth * phys_scale, ground_truth * phys_scale, fov_mask)
        fbp_img, _ = process_for_raps(fbp_pred * phys_scale, ground_truth * phys_scale, fov_mask)
        
        raps_dict = {}
        raps_dict["Ground Truth"] = compute_raps(gt_img)
        raps_dict["FBP (Analytical)"] = compute_raps(fbp_img)
        
        for name, model in models.items():
            pred = model(sinogram)
            # CRITICAL Q1 FIX: Dropped ref_mu parameter to bypass histogram tracking
            p_img, _ = process_for_raps(pred * phys_scale, ground_truth * phys_scale, fov_mask)
            raps_dict[name] = compute_raps(p_img)

    freqs = np.linspace(0, 0.5, len(raps_dict["Ground Truth"]))
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = {"Ground Truth": "k", "FBP (Analytical)": "gray", "FS-CNN (PAUM)": "#1f77b4", "FS-ViT (JotlasNet)": "#2ca02c", "FS-Net (Ours)": "#d62728"}
    styles = {"Ground Truth": "-", "FBP (Analytical)": ":", "FS-CNN (PAUM)": "--", "FS-ViT (JotlasNet)": "-.", "FS-Net (Ours)": "-"}
    widths = {"Ground Truth": 2.5, "FBP (Analytical)": 1.5, "FS-CNN (PAUM)": 1.75, "FS-ViT (JotlasNet)": 1.75, "FS-Net (Ours)": 2.22}
    
    for name in raps_dict.keys():
        ax.plot(freqs, raps_dict[name], color=colors[name], linestyle=styles[name], linewidth=widths[name], label=name)
    
    # Q1 JOURNAL FIX: Removed internal main text title. Captions go to LaTeX \caption.
    ax.set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=14, fontweight='bold')
    ax.set_ylabel(r"Spectral Power $\left[\log_{10}\right]$", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.5)
    
    min_power = np.min(raps_dict["Ground Truth"])
    ax.set_ylim(min_power - 0.5, 0.5)
    ax.legend(fontsize=11, loc='lower left', frameon=True, shadow=False, framealpha=0.95, edgecolor='lightgray')
    
    # --- RIGOROUS INSET AXES SPECIFICATION ---
    # Placed safely away from lower-left legend boundaries
    axins = ax.inset_axes([0.55, 0.52, 0.4, 0.42]) 
    for name in raps_dict.keys():
        axins.plot(freqs, raps_dict[name], color=colors[name], linestyle=styles[name], linewidth=widths[name])
    
    # Calculate a precise bounding envelope for the high-frequency target zone
    idx_03 = int(0.3 / 0.5 * len(freqs))
    all_inset_data = [raps_dict[name][idx_03:] for name in raps_dict.keys()]
    inset_min = np.min(all_inset_data) - 0.2
    inset_max = np.max(all_inset_data) + 0.2
    
    axins.set_xlim(0.3, 0.5)
    axins.set_ylim(inset_min, inset_max)
    
    # Q1 JOURNAL FIX: Add strict scientific ticks to the zoom block instead of plain blank lines
    axins.set_xticks([0.3, 0.4, 0.5])
    axins.set_xticklabels(["0.3", "0.4", "0.5"], fontsize=10)
    axins.tick_params(axis='y', labelsize=10)
    axins.grid(True, linestyle=':', alpha=0.5)
    
    # Tightly bounding indicators
    rect, connectors = ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.4)
    for connector in connectors:
        connector.set_linewidth(1.0)
        connector.set_color("gray")
    
    plt.tight_layout()
    plt.savefig("fig_raps_q1_compliant.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'fig_raps_q1_compliant.png'")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        generate_full_raps(LODOPAB_PATH, device=device)