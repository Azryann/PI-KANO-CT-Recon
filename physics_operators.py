import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExactParallelBeamRadon(nn.Module):
    def __init__(self, num_angles=1000, resolution=513, device='cuda'):
        """
        Initializes the exact discrete physics operator.
        Defaults are scaled for the LoDoPaB-CT dataset dimensions.
        """
        super().__init__()
        self.num_angles = num_angles
        self.resolution = resolution
        self.device = device
        
        # Define angles from 0 to Pi (180 degrees)
        self.theta = torch.linspace(0, math.pi, num_angles, device=device)
        
        # Precompute affine matrices for grid_sample rotation
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        zeros = torch.zeros_like(self.theta)
        
        # Build rotation matrices: [cos(t), -sin(t), 0; sin(t), cos(t), 0]
        self.affine_matrices = torch.stack([
            torch.stack([cos_t, -sin_t, zeros], dim=1),
            torch.stack([sin_t,  cos_t, zeros], dim=1)
        ], dim=1)

    def forward(self, x):
        """
        Forward Projection: A(x) -> Sinogram
        x shape: (Batch, Channels, Height, Width)
        """
        B, C, H, W = x.shape
        
        # Expand input to compute all angles simultaneously
        x_expanded = x.unsqueeze(1).expand(B, self.num_angles, C, H, W).reshape(B * self.num_angles, C, H, W)
        
        # Expand affine matrices for the batch
        matrices = self.affine_matrices.unsqueeze(0).expand(B, self.num_angles, 2, 3).reshape(B * self.num_angles, 2, 3)
        
        # Create rotated sampling grids
        grids = F.affine_grid(matrices, size=x_expanded.shape, align_corners=False)
        
        # Rotate the image batch
        rotated = F.grid_sample(x_expanded, grids, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # Project by summing along the ray direction (dim=2 for height)
        sinogram = rotated.sum(dim=2)
        
        # Reshape to standard sinogram format: (Batch, Channels, Angles, Detectors)
        sinogram = sinogram.view(B, self.num_angles, C, W)
        return sinogram

    def adjoint(self, y):
        """
        The Exact Discrete Adjoint: A^T(y) -> Image
        This uses autograd's Vector-Jacobian Product to guarantee zero adjoint mismatch.
        """
        # y shape is (Batch, Angles, Channels, Detectors) -> (B, 180, C, 128)
        # We need dummy_x to be (Batch, Channels, Resolution, Resolution) -> (B, C, 128, 128)
        batch_size = y.shape[0]
        channels = y.shape[2]  # <--- FIX: Grab channels from index 2, not index 1
        
        dummy_x = torch.zeros(
            (batch_size, channels, self.resolution, self.resolution), 
            requires_grad=True, 
            device=self.device
        )
        
        # Run the forward pass inside a grad context
        with torch.enable_grad():
            proj = self.forward(dummy_x)
            
            # The backward pass natively computes A^T * y
            proj.backward(y)
            
        # The gradient accumulated in dummy_x is the exact unfiltered backprojection
        return dummy_x.grad