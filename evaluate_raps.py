import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloaders import get_ct_dataloader
from physics import RadonPhysics

def compute_raps(image_np):
    """ Computes the 1D Radially Averaged Power Spectrum of a 2D image. """
    # 1. 2D FFT
    f = np.fft.fft2(image_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)**2
    
    # 2. Radial Averaging
    h, w = magnitude_spectrum.shape
    y, x = np.indices((h, w))
    center = (int(h/2), int(w/2))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    
    # Normalize and log scale
    radialprofile = radialprofile / radialprofile[0]
    return np.log10(radialprofile + 1e-8)

def plot_baseline_raps(data_path, device='cuda'):
    print("Computing RAPS for Ground Truth and FBP Baseline...")
    img_size, angles, detectors = 362, 1000, 513
    
    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    
    # Grab one slice
    for sino, gt in dataloader:
        sinogram = sino.to(device)
        ground_truth = gt.to(device)
        break
        
    # Generate FBP
    with torch.no_grad():
        fbp_pred = physics.adjoint(sinogram)
        
    gt_np = ground_truth.squeeze().cpu().numpy()
    fbp_np = fbp_pred.squeeze().cpu().numpy()
    
    raps_gt = compute_raps(gt_np)
    raps_fbp = compute_raps(fbp_np)
    
    freqs = np.linspace(0, 0.5, len(raps_gt)) # Normalized spatial frequency
    
    plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(freqs, raps_gt, 'k-', linewidth=2.5, label='Ground Truth')
    plt.plot(freqs, raps_fbp, 'gray', linestyle=':', linewidth=2, label='FBP (Baseline)')
    
    plt.title("Radially Averaged Power Spectrum (RAPS)", fontsize=14, fontweight='bold')
    plt.xlabel("Spatial Frequency (cycles/pixel)", fontsize=12)
    plt.ylabel("Log10 Power", fontsize=12)
    plt.xlim(0, 0.5)
    plt.ylim(-6, 1)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("fig_raps_baseline.png", dpi=300)
    print("SUCCESS: Saved 'fig_raps_baseline.png'")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    plot_baseline_raps(LODOPAB_PATH, device=device)