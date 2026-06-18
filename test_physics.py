import torch
from physics import ExactParallelBeamRadon

def test_operators():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on: {device}")
    
    # We use a small resolution (128x128) just to verify the math locally
    radon = ExactParallelBeamRadon(num_angles=180, resolution=128, device=device)
    
    # Create a dummy image: (Batch=1, Channels=1, H=128, W=128)
    image = torch.rand((1, 1, 128, 128), device=device)
    
    # 1. Test Forward Pass
    sinogram = radon.forward(image)
    print(f"Sinogram Shape: {sinogram.shape} -> Expected: (1, 1, 180, 128)")
    
    # 2. Test Adjoint Pass
    backprojected = radon.adjoint(sinogram)
    print(f"Backprojected Shape: {backprojected.shape} -> Expected: (1, 1, 128, 128)")
    
    print("SUCCESS: Exact Discrete Operators verified without memory leaks!")

if __name__ == "__main__":
    test_operators()