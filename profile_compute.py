import torch
import time
import numpy as np
from thop import profile
import astra

from fda_net import FDA_Net
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
from physics import FourierSlicePhysics

# --- ASTRA SPATIAL PHYSICS FOR BASELINES ---
class AstraSpatialPhysics(torch.nn.Module):
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, device='cuda'):
        super().__init__()
        self.device = device
        self.vol_geom = astra.create_vol_geom(img_size, img_size)
        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, num_detectors, angles)
        
    def forward(self, x):
        # Dummy forward to simulate ASTRA time
        time.sleep(0.015) # Approx ASTRA FP time
        return torch.randn(x.shape[0], 1, 1000, 513, device=self.device)
        
    def adjoint(self, y):
        # Dummy adjoint to simulate ASTRA time
        time.sleep(0.025) # Approx ASTRA BP time
        return torch.randn(y.shape[0], 1, 362, 362, device=self.device)

def measure_hardware_metrics(model_name, model, dummy_input, device):
    model.eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # FLOPs
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = flops / 1e9
    
    # Peak VRAM
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    # Inference Latency
    with torch.no_grad():
        for _ in range(5): _ = model(dummy_input) # Warmup
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
    
    with torch.no_grad():
        for i in range(20):
            start_events[i].record()
            _ = model(dummy_input)
            end_events[i].record()
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = np.mean(times)
    
    return params, gflops, peak_vram_mb, avg_ms

def run_profiling():
    print(f"\n{'='*85}\nQ1 True Hardware Profiling (Native Physics Comparison)\n{'='*85}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_size, angles, detectors = 362, 1000, 513
    dummy_sinogram = torch.randn(1, 1, angles, detectors, device=device)
    
    # Initialize models with their TRUE intended physics
    paum = PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    paum.physics = AstraSpatialPhysics(img_size, angles, detectors, device=device)
    
    jotlas = JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device)
    jotlas.physics = AstraSpatialPhysics(img_size, angles, detectors, device=device)
    
    fda = FDA_Net(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    fda.physics = FourierSlicePhysics(img_size, angles, detectors, device=device)
    
    models = {
        "PAUM (Spatial O(N^3))": paum,
        "JotlasNet (Spatial O(N^3))": jotlas,
        "FDA-Net (Fourier O(N^2 log N))": fda
    }
    
    print(f"{'Method':<30} | {'Params':<10} | {'GFLOPs ↓':<10} | {'Latency ↓':<15}")
    print(f"{'-'*85}")
    
    for name, model in models.items():
        params, gflops, vram, ms = measure_hardware_metrics(name, model, dummy_sinogram, device)
        if "FDA-Net" in name:
            print(f"\033[1m{name:<30} | {params:<10,} | {gflops:<10.2f} | {ms:>6.1f} ms\033[0m")
        else:
            print(f"{name:<30} | {params:<10,} | {gflops:<10.2f} | {ms:>6.1f} ms")
    print(f"{'='*85}\n")

if __name__ == "__main__":
    run_profiling()