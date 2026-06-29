import torch
import time
import numpy as np

# Import our models
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_neural_latency(model_name, model, dummy_x, dummy_grad, num_runs=100):
    """ 
    Isolates the Neural Architecture latency (bypassing the physics engine).
    This provides an apples-to-apples comparison of the learned regularizers.
    """
    model.eval()
    
    # Extract the neural update block based on the model type
    if model_name == "PAUM":
        neural_block = model.blocks[0]
        def run_step(): return neural_block(torch.cat([dummy_x, dummy_grad], dim=1))
        
    elif model_name == "JotlasNet":
        def run_step():
            feat = model.embed(torch.cat([dummy_x, dummy_grad], dim=1))
            B, C, H, W = feat.shape
            feat_flat = feat.view(B, C, -1).permute(0, 2, 1)
            feat_trans = model.transformer(feat_flat).permute(0, 2, 1).view(B, C, H, W)
            return model.de_embed(feat_trans)
            
    elif model_name == "PI-KINN":
        # PI-KINN requires the voltage state
        v_state = torch.zeros(dummy_x.shape[0], 32, dummy_x.shape[2], dummy_x.shape[3], device=dummy_x.device)
        def run_step():
            current_I = model.lifting(torch.cat([dummy_x, dummy_grad], dim=1))
            v_next = model.kinn_cell(current_I, v_state)
            return model.projection(v_next)

    # 1. GPU Warm-up
    with torch.no_grad():
        for _ in range(20):
            _ = run_step()
            
    torch.cuda.synchronize()
    
    # 2. Measure Latency
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    
    with torch.no_grad():
        for i in range(num_runs):
            start_events[i].record()
            _ = run_step()
            end_events[i].record()
            
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return np.mean(times), np.std(times)

def run_profiling():
    print(f"\n{'='*75}\nStarting Q1 Computational Profiling (Neural Architecture Isolation)\n{'='*75}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_size, angles, detectors = 362, 1000, 513
    
    # Dummy tensors for the image space (where the neural networks operate)
    dummy_x = torch.randn(1, 1, img_size, img_size, device=device)
    dummy_grad = torch.randn(1, 1, img_size, img_size, device=device)
    
    models = {
        "PAUM": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "PI-KINN": PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    print(f"{'Method':<20} | {'Parameters':<12} | {'Neural Update Latency (ms)':<25}")
    print(f"{'-'*75}")
    
    for name, model in models.items():
        params = count_parameters(model)
        avg_ms, std_ms = measure_neural_latency(name, model, dummy_x, dummy_grad)
        
        if "PI-KINN" in name:
            print(f"\033[1m{name:<20} | {params:<12,} | {avg_ms:>6.2f} ± {std_ms:>4.2f} ms\033[0m")
        else:
            print(f"{name:<20} | {params:<12,} | {avg_ms:>6.2f} ± {std_ms:>4.2f} ms")
            
    print(f"{'='*75}\n")
    print("Note: Latency reflects a single unrolled cascade update (excluding physics).")

if __name__ == "__main__":
    run_profiling()