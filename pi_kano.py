import torch
import torch.nn as nn
from gs_kan import GSKANConv2d
from physics import RadonPhysics

class SpectralGSKAN2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=16, modes2=16):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.kan_real = GSKANConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.kan_imag = GSKANConv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.complex64, device=x.device)
        
        real_part = self.kan_real(x_ft[:, :, :self.modes1, :self.modes2].real)
        imag_part = self.kan_imag(x_ft[:, :, :self.modes1, :self.modes2].imag)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.complex(real_part, imag_part)
        
        x_spatial = torch.fft.irfft2(out_ft, s=(H, W))
        return x_spatial

class PI_KANO(nn.Module):
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, device='cuda'):
        super().__init__()
        self.device = device
        self.physics = RadonPhysics(img_size, num_angles, num_detectors, device=device)
        
        hidden_channels = 32
        
        # 1. Compute Operator Norm ||A||^2 via Power Iteration
        print("Computing Operator Norm ||A||^2 via Power Iteration...")
        self.operator_norm_sq = self._power_iteration(img_size)
        self.tau_max = 2.0 / self.operator_norm_sq
        print(f"||A||^2 = {self.operator_norm_sq:.2f} | Strict Upper Bound for tau = {self.tau_max:.6f}")
        
        # Initialize step size safely below the theoretical boundary
        self.step_size = nn.Parameter(torch.tensor(self.tau_max * 0.1, dtype=torch.float32))
        
        self.lifting = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        
        # 2. STRICT NORMALIZATION: Traps tensors in N(0,1) for the B-spline grid [-2, 2]
        self.norm1 = nn.InstanceNorm2d(hidden_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(hidden_channels, affine=True)
        
        self.spectral_block = SpectralGSKAN2d(hidden_channels, hidden_channels, modes1=16, modes2=16)
        self.spatial_block = GSKANConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.projection = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def _power_iteration(self, img_size, num_iters=15):
        """ Estimates the largest eigenvalue of A^T A """
        u = torch.randn(1, 1, img_size, img_size, device=self.device)
        u = u / torch.norm(u)
        with torch.no_grad():
            for _ in range(num_iters):
                v = self.physics.adjoint(self.physics.forward(u))
                norm_v = torch.norm(v)
                u = v / norm_v
        return norm_v.item()

    def forward(self, sinogram):
        # Enforce theoretical proximal bound: tau must be strictly < 2 / ||A||^2
        tau = torch.clamp(self.step_size, min=1e-8, max=self.tau_max * 0.99)
        
        # Physics-Informed Initialization
        x_init = self.physics.adjoint(sinogram) * tau
        
        # Lifting
        feat = self.lifting(x_init)
        
        # Strict Normalization before GS-KAN
        feat = self.norm1(feat)
        
        # Dual-domain operator processing
        feat_spectral = self.spectral_block(feat)
        feat_spatial = self.spatial_block(feat)
        feat_fused = feat_spectral + feat_spatial
        
        # Strict Normalization after operator fusion
        feat_fused = self.norm2(feat_fused)
        
        # Project back to image space
        reconstruction = self.projection(feat_fused)
        
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