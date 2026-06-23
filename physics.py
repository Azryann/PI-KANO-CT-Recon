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
        
        # PyTorch linspace fix: Create angles from 0 to Pi (excluding Pi)
        end_angle = np.pi * (num_angles - 1) / num_angles
        theta = torch.linspace(0, end_angle, num_angles, device=device)
        omega = torch.linspace(-1.0, 1.0, num_detectors, device=device)
        
        # FIX: Use 'ij' indexing so output is strictly (Angles, Detectors)
        Theta, Omega = torch.meshgrid(theta, omega, indexing='ij')
        
        kx = Omega * torch.cos(Theta)
        ky = Omega * torch.sin(Theta)
        
        # Grid shape: (1, Angles, Detectors, 2) -> (1, 1000, 513, 2)
        self.polar_grid = torch.stack([kx, ky], dim=-1).unsqueeze(0)
        print("Fourier-Slice Edge-Physics Initialized. (ASTRA bypassed).")

    def forward(self, x):
        B, C, H, W = x.shape
        X_freq = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        
        X_real, X_imag = X_freq.real, X_freq.imag
        grid = self.polar_grid.expand(B, -1, -1, -1)
        
        polar_real = F.grid_sample(X_real, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        polar_imag = F.grid_sample(X_imag, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        polar_freq = torch.complex(polar_real, polar_imag)
        polar_freq = torch.fft.ifftshift(polar_freq, dim=-1)
        
        # Output shape is now strictly (B, C, 1000, 513)
        return torch.fft.ifft(polar_freq, dim=-1).real

    def adjoint(self, y):
        B, C, Angles, Detectors = y.shape
        Y_freq = torch.fft.fftshift(torch.fft.fft(y, dim=-1), dim=-1)
        
        omega = torch.abs(torch.linspace(-1.0, 1.0, Detectors, device=y.device))
        Y_filtered = Y_freq * omega.view(1, 1, 1, -1)
        y_filtered = torch.fft.ifft(torch.fft.ifftshift(Y_filtered, dim=-1), dim=-1).real
        
        # Create reconstruction grid
        x_grid, y_grid = torch.meshgrid(
            torch.linspace(-1, 1, self.img_size, device=y.device),
            torch.linspace(-1, 1, self.img_size, device=y.device),
            indexing='ij'
        )
        
        end_angle = np.pi * (Angles - 1) / Angles
        theta = torch.linspace(0, end_angle, Angles, device=y.device)
        reconstruction = torch.zeros(B, C, self.img_size, self.img_size, device=y.device)
        
        for i in range(Angles):
            t = x_grid * torch.cos(theta[i]) + y_grid * torch.sin(theta[i])
            t_norm = t / 1.414 
            
            # Grid shape: (B, img_size, img_size, 2)
            grid = torch.stack([t_norm, torch.zeros_like(t_norm)], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            
            # Proj shape: (B, C, 1, Detectors)
            proj = y_filtered[:, :, i, :].unsqueeze(2) 
            
            # Backproj shape: (B, C, img_size, img_size)
            backproj = F.grid_sample(proj, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            reconstruction += backproj
            
        return reconstruction * (np.pi / Angles)

# ALIAS for compatibility with existing scripts
RadonPhysics = FourierSlicePhysics