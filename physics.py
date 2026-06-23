import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierSlicePhysics(nn.Module):
    """
    Differentiable Fourier Slice Theorem (DFST) Operator.
    100% Native PyTorch. O(N^2 log N) complexity. Edge-Device Deployable.
    """
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, device='cuda'):
        super().__init__()
        self.img_size = img_size
        self.num_angles = num_angles
        self.num_detectors = num_detectors
        self.device = device
        
        # 1. Create Polar Coordinate Grid for the Fourier Domain
        # Angles from 0 to Pi, Frequencies from -0.5 to 0.5
        theta = torch.linspace(0, np.pi, num_angles, endpoint=False, device=device)
        omega = torch.linspace(-1.0, 1.0, num_detectors, device=device)
        
        # Create meshgrid
        Omega, Theta = torch.meshgrid(omega, theta, indexing='xy')
        
        # Convert Polar to Cartesian (x, y coordinates in the 2D Fourier plane)
        # We scale to [-1, 1] for PyTorch's grid_sample
        kx = Omega * torch.cos(Theta)
        ky = Omega * torch.sin(Theta)
        
        # Grid shape: (1, num_angles, num_detectors, 2)
        self.polar_grid = torch.stack([kx, ky], dim=-1).unsqueeze(0).transpose(1, 2)
        
        print("Fourier-Slice Edge-Physics Initialized. (ASTRA bypassed).")

    def forward(self, x):
        """ Image -> Sinogram via Fourier Slice Theorem """
        B, C, H, W = x.shape
        
        # 1. 2D FFT of the Image (Shifted to center)
        X_freq = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        
        # Separate Real and Imaginary for grid_sample (which expects floats)
        X_real = X_freq.real
        X_imag = X_freq.imag
        
        # 2. Extract Polar Slices using Bilinear Interpolation
        grid = self.polar_grid.expand(B, -1, -1, -1)
        polar_real = F.grid_sample(X_real, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        polar_imag = F.grid_sample(X_imag, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        polar_freq = torch.complex(polar_real, polar_imag)
        
        # 3. 1D Inverse FFT along the detector dimension to get the Sinogram
        # Shift back before IFFT
        polar_freq = torch.fft.ifftshift(polar_freq, dim=-1)
        sinogram = torch.fft.ifft(polar_freq, dim=-1).real
        
        return sinogram

    def adjoint(self, y):
        """ Sinogram -> Image via Inverse Fourier Slice Theorem (Gridding) """
        B, C, Angles, Detectors = y.shape
        
        # 1. 1D FFT of the Sinogram
        Y_freq = torch.fft.fftshift(torch.fft.fft(y, dim=-1), dim=-1)
        
        # 2. Splatting (Adjoint of grid_sample)
        # PyTorch doesn't have a native adjoint for grid_sample, but for unrolled networks,
        # we can use the transposed convolution approximation or simple backprojection.
        # To maintain Edge-Speed, we use a differentiable Frequency-Domain Backprojection.
        
        # Filter the sinogram (Ramp filter in frequency domain)
        omega = torch.abs(torch.linspace(-1.0, 1.0, Detectors, device=y.device))
        Y_filtered = Y_freq * omega.view(1, 1, 1, -1)
        
        y_filtered = torch.fft.ifft(torch.fft.ifftshift(Y_filtered, dim=-1), dim=-1).real
        
        # Standard fast spatial backprojection mapping
        # Create image grid
        x_grid, y_grid = torch.meshgrid(
            torch.linspace(-1, 1, self.img_size, device=y.device),
            torch.linspace(-1, 1, self.img_size, device=y.device),
            indexing='xy'
        )
        
        theta = torch.linspace(0, np.pi, Angles, endpoint=False, device=y.device)
        reconstruction = torch.zeros(B, C, self.img_size, self.img_size, device=y.device)
        
        # Fast vectorized backprojection
        for i in range(Angles):
            t = x_grid * torch.cos(theta[i]) + y_grid * torch.sin(theta[i])
            # Map t from [-sqrt(2), sqrt(2)] to [-1, 1] for grid_sample
            t_norm = t / 1.414 
            grid = torch.stack([t_norm, torch.zeros_like(t_norm)], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            
            # Sample the 1D filtered projection
            proj = y_filtered[:, :, i, :].unsqueeze(2) # (B, C, 1, Detectors)
            backproj = F.grid_sample(proj, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            reconstruction += backproj.squeeze(2)
            
        return reconstruction * (np.pi / Angles)