import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dataloaders import get_ct_dataloader
from physics import RadonPhysics
from fda_net import FDA_Net
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate

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
    
    radialprofile = radialprofile / radialprofile[0]
    return np.log10(radialprofile + 1e-8)

def generate_full_raps(data_path, device='cuda'):
    print("Generating Q1 RAPS Plot with High-Frequency Inset...")
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
    
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    
    models = {
        "PAUM": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "FDA-Net (Ours)": FDA_Net(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    for name, model in models.items():
        ckpt_name = name.split(" ")[0]
        ckpt_path = f"{ckpt_name}_subset_BEST.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model_state'])
            model.eval()

    for i, (sino, gt) in enumerate(dataloader):
        if i == 15: 
            sinogram = sino.to(device) / phys_scale
            ground_truth = gt.to(device) / phys_scale
            break

    with torch.no_grad():
        fbp_pred = physics.adjoint(sinogram)
        raps_dict = {}
        raps_dict["Ground Truth"] = compute_raps(ground_truth.squeeze().cpu().numpy())
        raps_dict["FBP"] = compute_raps(fbp_pred.squeeze().cpu().numpy())
        
        for name, model in models.items():
            pred = model(sinogram)
            raps_dict[name] = compute_raps(pred.squeeze().cpu().numpy())

    freqs = np.linspace(0, 0.5, len(raps_dict["Ground Truth"]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = {"Ground Truth": "k", "FBP": "gray", "PAUM": "#1f77b4", "JotlasNet": "#2ca02c", "FDA-Net (Ours)": "#d62728"}
    styles = {"Ground Truth": "-", "FBP": ":", "PAUM": "--", "JotlasNet": "-.", "FDA-Net (Ours)": "-"}
    widths = {"Ground Truth": 3, "FBP": 2, "PAUM": 2, "JotlasNet": 2, "FDA-Net (Ours)": 2.5}
    
    for name in raps_dict.keys():
        ax.plot(freqs, raps_dict[name], color=colors[name], linestyle=styles[name], linewidth=widths[name], label=name)
    
    ax.set_title("Radially Averaged Power Spectrum (RAPS)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=14)
    ax.set_ylabel("Log10 Power", fontsize=14)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-6, 1)
    ax.legend(fontsize=12, loc='lower left', frameon=True, shadow=True)
    
    # --- INSET BOX (High-Frequency Zoom) ---
    axins = ax.inset_axes([0.6, 0.55, 0.35, 0.4]) # [x, y, width, height]
    for name in raps_dict.keys():
        axins.plot(freqs, raps_dict[name], color=colors[name], linestyle=styles[name], linewidth=widths[name])
    
    # Zoom in on the high-frequency tail
    axins.set_xlim(0.3, 0.5)
    axins.set_ylim(-5.5, -2.5)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_title("High-Frequency Detail", fontsize=10)
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    plt.tight_layout()
    plt.savefig("fig_raps_full.png", dpi=300)
    print("SUCCESS: Saved 'fig_raps_full.png'")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        generate_full_raps(LODOPAB_PATH, device=device)