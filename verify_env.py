import torch

def verify_cuda():
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / (1024**3)
        print(f"CUDA is AVAILABLE.")
        print(f"Device Name: {props.name}")
        print(f"Total VRAM: {vram_gb:.2f} GB")
        
        if vram_gb < 5.0:
            print("STATUS: Local Prototyping Environment Detected (<= 4GB VRAM).")
            print("ACTION: Will use dummy tensors and minimal batch sizes.")
        else:
            print("STATUS: Cloud/Kaggle Environment Detected (> 4GB VRAM).")
            print("ACTION: Cleared for full-scale training.")
    else:
        print("CUDA is NOT available. Running on CPU.")

if __name__ == "__main__":
    verify_cuda()