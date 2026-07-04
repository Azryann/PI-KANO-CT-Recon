import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    print("Generating Full Q1 RAPS Plot...")
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
        ckpt_path = f"{ckpt_name}_subset_BEST.pth" # Using the BEST checkpoints
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
            model.eval()

    for i, (sino, gt) in enumerate(dataloader):
        if i == 15: # Use the same good slice we used for visuals
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

    # --- PLOTTING ---
    freqs = np.linspace(0, 0.5, len(raps_dict["Ground Truth"]))
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(freqs, raps_dict["Ground Truth"], 'k-', linewidth=3, label='Ground Truth')
    plt.plot(freqs, raps_dict["FBP"], 'gray', linestyle=':', linewidth=2, label='FBP (Baseline)')
    plt.plot(freqs, raps_dict["PAUM"], '#1f77b4', linestyle='--', linewidth=2, label='PAUM (CNN)')
    plt.plot(freqs, raps_dict["JotlasNet"], '#2ca02c', linestyle='-.', linewidth=2, label='JotlasNet (ViT)')
    plt.plot(freqs, raps_dict["FDA-Net (Ours)"], '#d62728', linestyle='-', linewidth=2.5, label='FDA-Net (Ours)')
    
    plt.title("Radially Averaged Power Spectrum (RAPS)", fontsize=16, fontweight='bold')
    plt.xlabel("Spatial Frequency (cycles/pixel)", fontsize=14)
    plt.ylabel("Log10 Power", fontsize=14)
    plt.xlim(0, 0.5)
    plt.ylim(-6, 1)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig("fig_raps_full.png", dpi=300)
    print("SUCCESS: Saved 'fig_raps_full.png'")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        generate_full_raps(LODOPAB_PATH, device=device)