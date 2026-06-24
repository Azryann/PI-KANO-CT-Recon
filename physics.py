import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierSlicePhysics(nn.Module):
    """
    Differentiable Fourier Slice Theorem (DFST) Operator.
    Fully Vectorized 1D Interpolation. 100% Native PyTorch. Edge-Device Deployable.
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
        self.polar_grid = torch.stack([kx, ky], dim=-1).unsqueeze(0)
        
        # ==========================================
        # 2. PRE-COMPUTE ADJOINT GRID (Cartesian -> Polar)
        # ==========================================
        x_grid, y_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, img_size, device=device),
            torch.linspace(-1.0, 1.0, img_size, device=device),
            indexing='xy'
        )
        
        x_flat = x_grid.reshape(-1)
        y_flat = y_grid.reshape(-1)
        
        # Shape: (Angles, N_Pixels)
        t = x_flat.unsqueeze(0) * torch.cos(theta).unsqueeze(1) + y_flat.unsqueeze(0) * torch.sin(theta).unsqueeze(1)
        
        # Save normalized coordinates for the Batched 1D interpolation
        self.t_norm = t / 1.41421356  
        
        # Pre-compute ramp filter
        omega_filter = torch.abs(torch.linspace(-1.0, 1.0, num_detectors, device=device))
        self.omega_filter = omega_filter.view(1, 1, 1, -1)
        
        print("Fully Vectorized 1D Fourier-Slice Physics Initialized. (No Blurring!).")

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
        
        # 2. Exact Batched 1D Vectorized Backprojection
        # Fold Angles into Batch dimension to prevent cross-angle interpolation blurring
        y_reshaped = y_filtered.view(B * Angles, C, 1, Detectors)
        
        # Expand pre-computed grid to match (B * Angles)
        t_batch = self.t_norm.unsqueeze(0).expand(B, -1, -1).reshape(B * Angles, -1)
        y_coords = torch.zeros_like(t_batch)
        
        # Grid shape: (B * Angles, 1, N_Pixels, 2)
        grid = torch.stack([t_batch, y_coords], dim=-1).unsqueeze(1)
        
        # Pure 1D interpolation along the detector array
        sampled = F.grid_sample(y_reshaped, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 3. Sum over angles and reshape back to images
        sampled = sampled.view(B, Angles, C, self.img_size, self.img_size)
        reconstruction = sampled.sum(dim=1) * (np.pi / Angles)
            
        return reconstruction

# ALIAS
RadonPhysics = FourierSlicePhysics