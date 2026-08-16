import time
import torch
import numpy as np
import onnxruntime as ort
import psutil
import os

from fda_net import FDA_Net

def benchmark_mobile():
    # 1. Configuration
    device = "cpu"
    img_size, angles, detectors = 362, 1000, 513
    num_warmup = 10
    num_test = 100  # 100 slices is statistically stable
    onnx_path = "fs_net_mobile.onnx"

    print("Initializing FS-Net PyTorch Model...")
    # Initialize your model with the exact parameters from your notebook
    model = FDA_Net(img_size, angles, detectors, num_cascades=3, device=device)
    model.eval()

    # 2. Export to ONNX
    print(f"Exporting model to {onnx_path}...")
    dummy_input = torch.randn(1, 1, angles, detectors, dtype=torch.float32)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['sinogram'],
        output_names=['reconstruction']
    )

    # 3. Initialize ONNX Runtime
    print("Loading ONNX Runtime Session...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4  # Typical for mobile big.LITTLE ARM cores
    
    # Attempt to use Android Neural Networks API if available, otherwise fallback to ARM CPU
    providers = ['NnapiExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, session_options, providers=providers)
    
    input_name = session.get_inputs()[0].name
    test_input = np.random.randn(1, 1, angles, detectors).astype(np.float32)

    # 4. Warmup Phase
    print(f"Running {num_warmup} warmup iterations (Discarding inflated latencies)...")
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: test_input})

    # 5. Benchmarking Phase
    print(f"Running {num_test} test iterations for stable latency...")
    latencies = []
    for _ in range(num_test):
        start_time = time.time()
        _ = session.run(None, {input_name: test_input})
        latencies.append((time.time() - start_time) * 1000) # Convert to ms

    # 6. Resource Profiling
    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / (1024 * 1024)

    # 7. Print Table Data
    print("\n" + "="*50)
    print("TABLE: ON-DEVICE INFERENCE VALIDATION (SAMSUNG F17)")
    print("="*50)
    print(f"Device Name      : Samsung F17 (via Termux)")
    print(f"Runtime Used     : ONNX Runtime {ort.__version__} ({session.get_providers()[0]})")
    print(f"Batch Size       : 1")
    print(f"Input Resolution : {img_size}x{img_size} (Sinogram: {angles}x{detectors})")
    print(f"Mean Latency     : {np.mean(latencies):.2f} ± {np.std(latencies):.2f} ms")
    print(f"Peak RAM         : {peak_ram_mb:.2f} MB")
    print(f"Power Draw (W)   : Unavailable via Termux (N/A)")
    print(f"Test Slices Used : {num_test}")
    print("="*50)

if __name__ == "__main__":
    benchmark_mobile()