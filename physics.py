import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierSlicePhysics(nn.Module):
    """
    Differentiable Fourier Slice Theorem (DFST) Operator.
    Fully Vectorized. 100% Native PyTorch. Edge-Device Deployable.
    """
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, device='cuda'):
        super().__init__()
        self.img_size = img_size
        self.num_angles = num_angles
        self.num_detectors = num_detectors
        self.device = device
        
        # ==========================================
        # 1. PRE-COMPUTE FORWARD GRID (Polar -> Cartesian)
        # ==========================================
        end_angle = np.pi * (num_angles - 1) / num_angles
        theta = torch.linspace(0, end_angle, num_angles, device=device)
        omega = torch.linspace(-1.0, 1.0, num_detectors, device=device)
        
        Theta, Omega = torch.meshgrid(theta, omega, indexing='ij')
        kx = Omega * torch.cos(Theta)
        ky = Omega * torch.sin(Theta)
        
        # Grid shape: (1, Angles, Detectors, 2)
        self.polar_grid = torch.stack([kx, ky], dim=-1).unsqueeze(0)
        
        # ==========================================
        # 2. PRE-COMPUTE ADJOINT GRID (Cartesian -> Polar)
        # ==========================================
        x_grid, y_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, img_size, device=device),
            torch.linspace(-1.0, 1.0, img_size, device=device),
            indexing='xy'
        )
        
        # Flatten spatial dimensions to vectorize the 1000 angles
        x_flat = x_grid.reshape(-1)
        y_flat = y_grid.reshape(-1)
        
        # Compute projection coordinates for all angles and all pixels simultaneously
        # Shape: (Angles, N_Pixels)
        t = x_flat.unsqueeze(0) * torch.cos(theta).unsqueeze(1) + y_flat.unsqueeze(0) * torch.sin(theta).unsqueeze(1)
        t_norm = t / 1.41421356  # Normalize to [-1, 1]
        
        # Y-coordinate in grid_sample represents the Angle index mapped to [-1, 1]
        angle_y = torch.linspace(-1.0, 1.0, num_angles, device=device).unsqueeze(1).expand(num_angles, img_size * img_size)
        
        # Grid shape: (1, Angles, N_Pixels, 2)
        self.bp_grid = torch.stack([t_norm, angle_y], dim=-1).unsqueeze(0)
        
        # Pre-compute ramp filter
        omega_filter = torch.abs(torch.linspace(-1.0, 1.0, num_detectors, device=device))
        self.omega_filter = omega_filter.view(1, 1, 1, -1)
        
        print("Fully Vectorized Fourier-Slice Physics Initialized. (Zero Python Loops).")

    def forward(self, x):
        B, C, H, W = x.shape
        X_freq = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        
        grid = self.polar_grid.expand(B, -1, -1, -1)
        
        polar_real = F.grid_sample(X_freq.real, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        polar_imag = F.grid_sample(X_freq.imag, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        polar_freq = torch.complex(polar_real, polar_imag)
        polar_freq = torch.fft.ifftshift(polar_freq, dim=-1)
        
        return torch.fft.ifft(polar_freq, dim=-1).real

    def adjoint(self, y):
        B, C, Angles, Detectors = y.shape
        
        # 1. 1D FFT & Ramp Filter
        Y_freq = torch.fft.fftshift(torch.fft.fft(y, dim=-1), dim=-1)
        Y_filtered = Y_freq * self.omega_filter
        y_filtered = torch.fft.ifft(torch.fft.ifftshift(Y_filtered, dim=-1), dim=-1).real
        
        # 2. Vectorized Backprojection (A single GPU call replaces 1000 loops!)
        grid = self.bp_grid.expand(B, -1, -1, -1)
        
        # Sample shape: (B, C, Angles, N_Pixels)
        sampled = F.grid_sample(y_filtered, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 3. Sum over angles and reshape
        reconstruction = sampled.sum(dim=2)  # Shape: (B, C, N_Pixels)
        reconstruction = reconstruction.view(B, C, self.img_size, self.img_size) * (np.pi / Angles)
            
        return reconstruction

# ALIAS for compatibility with existing scripts
RadonPhysics = FourierSlicePhysics