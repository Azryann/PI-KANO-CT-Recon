import torch
import time
import numpy as np

# Import our models
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, dummy_input, device, num_runs=100):
    model.eval()
    
    # Q1 FIX: JIT Compile the model to fuse the custom Kirchhoff ODE kernels
    # This ensures a fair hardware comparison against highly-optimized standard CNNs
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print("Torch compile failed, falling back to eager mode.")
        compiled_model = model
    
    # 1. GPU Warm-up (Crucial for accurate timing and JIT compilation)
    with torch.no_grad():
        for _ in range(20):
            _ = compiled_model(dummy_input)
            
    torch.cuda.synchronize()
    
    # 2. Measure Latency
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    
    with torch.no_grad():
        for i in range(num_runs):
            start_events[i].record()
            _ = compiled_model(dummy_input)
            end_events[i].record()
            
    torch.cuda.synchronize()
    
    # Calculate average time in milliseconds
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    
    return avg_time_ms, std_time_ms

def run_profiling():
    print(f"\n{'='*65}\nStarting Q1 Computational Profiling (JIT Compiled)\n{'='*65}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_size, angles, detectors = 362, 1000, 513
    dummy_sinogram = torch.randn(1, 1, angles, detectors, device=device)
    
    models = {
        "PAUM (CNN Baseline)": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet (Transformer)": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "PI-KINN (Ours)": PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    print(f"{'Method':<25} | {'Parameters':<12} | {'Inference Time (ms/slice)':<25}")
    print(f"{'-'*65}")
    
    for name, model in models.items():
        params = count_parameters(model)
        avg_ms, std_ms = measure_inference_time(model, dummy_sinogram, device)
        
        if "Ours" in name:
            print(f"\033[1m{name:<25} | {params:<12,} | {avg_ms:>6.1f} ± {std_ms:>4.1f} ms\033[0m")
        else:
            print(f"{name:<25} | {params:<12,} | {avg_ms:>6.1f} ± {std_ms:>4.1f} ms")
            
    print(f"{'='*65}\n")

if __name__ == "__main__":
    run_profiling()