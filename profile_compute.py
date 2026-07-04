import torch
import time
import numpy as np
from thop import profile

from fda_net import FDA_Net
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate

def measure_hardware_metrics(model_name, model, dummy_input, device):
    model.eval()
    
    # 1. Parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. FLOPs (Using THOP)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = flops / 1e9
    
    # 3. Peak VRAM
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    # 4. Inference Latency
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input) # Warmup
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    
    with torch.no_grad():
        for i in range(50):
            start_events[i].record()
            _ = model(dummy_input)
            end_events[i].record()
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = np.mean(times)
    
    return params, gflops, peak_vram_mb, avg_ms

def run_profiling():
    print(f"\n{'='*85}\nQ1 Industry-Grade Hardware Profiling (Edge-AI Metrics)\n{'='*85}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_size, angles, detectors = 362, 1000, 513
    dummy_sinogram = torch.randn(1, 1, angles, detectors, device=device)
    
    models = {
        "PAUM (CNN)": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet (Transformer)": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "FDA-Net (Ours)": FDA_Net(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    print(f"{'Method':<22} | {'Params':<10} | {'GFLOPs ↓':<10} | {'Peak VRAM ↓':<15} | {'Latency ↓':<15}")
    print(f"{'-'*85}")
    
    for name, model in models.items():
        params, gflops, vram, ms = measure_hardware_metrics(name, model, dummy_sinogram, device)
        if "Ours" in name:
            print(f"\033[1m{name:<22} | {params:<10,} | {gflops:<10.2f} | {vram:<10.1f} MB | {ms:>6.1f} ms\033[0m")
        else:
            print(f"{name:<22} | {params:<10,} | {gflops:<10.2f} | {vram:<10.1f} MB | {ms:>6.1f} ms")
    print(f"{'='*85}\n")

if __name__ == "__main__":
    run_profiling()