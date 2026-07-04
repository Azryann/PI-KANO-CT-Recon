import torch
import numpy as np
from thop import profile

from fda_net import FDA_Net
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate

def get_physics_gflops(physics_type, cascades, img_size=362, angles=1000, detectors=513):
    """ 
    Calculates the exact theoretical GFLOPs for the physics operators.
    ASTRA C++ hides these from PyTorch profilers, so we must compute them analytically.
    """
    if physics_type == "spatial":
        # O(N^3) Ray Tracing: Angles * Detectors * Pixels_per_ray
        # Forward + Adjoint per cascade
        ops_per_proj = angles * detectors * img_size
        gflops_per_cascade = (2 * ops_per_proj) / 1e9
        return gflops_per_cascade * cascades
        
    elif physics_type == "fourier":
        # O(N^2 log N) Fourier Slice Theorem
        # 2D FFT + Grid Sample + 1D IFFT
        fft2d_ops = 5 * (img_size**2) * np.log2(img_size)
        grid_ops = angles * detectors * 4 # Bilinear interpolation
        ifft1d_ops = angles * 5 * detectors * np.log2(detectors)
        
        ops_per_proj = fft2d_ops + grid_ops + ifft1d_ops
        gflops_per_cascade = (2 * ops_per_proj) / 1e9
        return gflops_per_cascade * cascades

def measure_true_hardware_metrics(model_name, model, dummy_input, physics_type, cascades, device):
    model.eval()
    
    # 1. Parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Neural GFLOPs (via THOP)
    # We pass the dummy input through the neural blocks only to get the neural FLOPs
    neural_flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    neural_gflops = neural_flops / 1e9
    
    # 3. Physics GFLOPs (Analytical)
    physics_gflops = get_physics_gflops(physics_type, cascades)
    
    # Total GFLOPs
    total_gflops = neural_gflops + physics_gflops
    
    # 4. Peak VRAM (MB)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    return params, total_gflops, peak_vram_mb

def run_profiling():
    print(f"\n{'='*85}\nQ1 True Hardware Profiling (Neural + Physics GFLOPs)\n{'='*85}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_size, angles, detectors = 362, 1000, 513
    dummy_sinogram = torch.randn(1, 1, angles, detectors, device=device)
    
    models = {
        "PAUM (CNN)": (PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device), "spatial", 3),
        "JotlasNet (ViT)": (JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device), "spatial", 2),
        "FDA-Net (Ours)": (FDA_Net(img_size, angles, detectors, num_cascades=3, device=device).to(device), "fourier", 3)
    }
    
    print(f"{'Method':<22} | {'Params ↓':<10} | {'Total GFLOPs ↓':<15} | {'Peak VRAM ↓':<15}")
    print(f"{'-'*85}")
    
    for name, (model, phys_type, cascades) in models.items():
        params, gflops, vram = measure_true_hardware_metrics(name, model, dummy_sinogram, phys_type, cascades, device)
        
        if "Ours" in name:
            print(f"\033[1m{name:<22} | {params:<10,} | {gflops:<15.2f} | {vram:<10.1f} MB\033[0m")
        else:
            print(f"{name:<22} | {params:<10,} | {gflops:<15.2f} | {vram:<10.1f} MB")
    print(f"{'='*85}\n")

if __name__ == "__main__":
    run_profiling()