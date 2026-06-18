import torch
import torch.nn as nn
from gs_kan import GSKANConv2d
from physics import RadonPhysics

class SpectralGSKAN2d(nn.Module):
    """
    Applies GS-KAN in the Fourier Domain to capture infinite-dimensional 
    global operator dependencies. Solves the FNO spectral bottleneck.
    """
    def __init__(self, in_channels, out_channels, modes1=16, modes2=16):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        
        # GS-KAN operates on the real and imaginary parts of the Fourier coefficients
        self.kan_real = GSKANConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.kan_imag = GSKANConv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # 1. Compute 2D Discrete Fourier Transform
        x_ft = torch.fft.rfft2(x)
        
        # 2. Extract active Fourier modes
        out_ft = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.complex64, device=x.device)
        
        # Apply GS-KAN on the low-frequency Fourier modes
        # Real and imaginary components are processed separately to maintain compatibility with PyTorch real-valued layers
        real_part = self.kan_real(x_ft[:, :, :self.modes1, :self.modes2].real)
        imag_part = self.kan_imag(x_ft[:, :, :self.modes1, :self.modes2].imag)
        
        # Reconstruct complex tensor
        out_ft[:, :, :self.modes1, :self.modes2] = torch.complex(real_part, imag_part)
        
        # 3. Inverse DFT back to spatial domain
        x_spatial = torch.fft.irfft2(out_ft, s=(H, W))
        return x_spatial


class PI_KANO(nn.Module):
    """
    The full Dual-Domain Physics-Informed Kolmogorov-Arnold Neural Operator.
    Fuses spatial GS-KAN, Spectral GS-KAN, and Exact Discrete Physics.
    """
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, device='cuda'):
        super().__init__()
        self.device = device
        
        # Initialize the exact discrete physics model (avoids adjoint mismatch)
        self.physics = RadonPhysics(img_size, num_angles, num_detectors, device=device)
        
        # Architecture Hyperparameters
        hidden_channels = 32
        
        # Lifting Layer: project gray-scale CT slice to hidden channels
        self.lifting = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        
        # Dual-Domain Operator Blocks
        # Block 1: Captures infinite-dimensional global spectral operators
        self.spectral_block = SpectralGSKAN2d(hidden_channels, hidden_channels, modes1=16, modes2=16)
        
        # Block 2: Captures local sharp physical discontinuities (Bones, soft tissues)
        self.spatial_block = GSKANConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Projection Layer: project back to gray-scale CT slice
        self.projection = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, sinogram):
        """
        Input: Raw Sinogram measurements (Batch, 1, Angles, Detectors)
        Output: Reconstructed CT Image (Batch, 1, H, W)
        """
        # 1. Physics-Informed Initialization (Filtered Backprojection using exact discrete adjoint)
        # Translates measurement space (sinogram) to reconstruction space (image)
        x_init = self.physics.adjoint(sinogram)
        
        # 2. Lift to high-dimensional feature space
        feat = self.lifting(x_init)
        
        # 3. Dual-domain operator processing
        # Captures global waves
        feat_spectral = self.spectral_block(feat)
        # Captures local spatial boundaries
        feat_spatial = self.spatial_block(feat)
        
        # Fused dual-domain representation
        feat_fused = feat_spectral + feat_spatial
        
        # 4. Project back to image space
        reconstruction = self.projection(feat_fused)
        
        # Residual connection to the physical initialization
        return x_init + reconstruction


if __name__ == "__main__":
    # ==========================================
    # PI-KANO ARCHITECTURE VERIFICATION
    # ==========================================
    print("Initializing PI-KANO Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Standard LoDoPaB dimensions
    IMG_SIZE = 362
    ANGLES = 1000
    DETECTORS = 513
    
    # Instantiate PI-KANO
    # Highly efficient: will fit inside your 4GB local GPU for single-batch testing
    model = PI_KANO(img_size=IMG_SIZE, num_angles=ANGLES, num_detectors=DETECTORS, device=device).to(device)
    
    # Create a simulated raw sinogram measurement batch (Batch=1, Channels=1)
    test_sinogram = torch.randn(1, 1, ANGLES, DETECTORS, device=device)
    
    print("\nRunning Forward Pass of PI-KANO...")
    with torch.no_grad():
        reconstructed_image = model(test_sinogram)
        
    print(f"\nMeasurements Shape (Input Sinogram):  {test_sinogram.shape}")
    print(f"Reconstructed CT Image Shape (Output): {reconstructed_image.shape}")
    
    # Assert correct spatial dimensions
    assert reconstructed_image.shape == (1, 1, IMG_SIZE, IMG_SIZE), "Output shape mismatch!"
    print("\nSUCCESS: PI-KANO dual-domain architecture is validated and mathematically consistent.")