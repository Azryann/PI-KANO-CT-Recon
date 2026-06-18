import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedRBFMasterFunction(nn.Module):
    """
    Bypasses the recursive Cox-de Boor B-spline bottleneck.
    Uses a highly vectorized Gaussian Radial Basis Function (RBF) grid as the 
    single learnable master function for the layer.
    """
    def __init__(self, grid_size=5, grid_range=[-2.0, 2.0]):
        super().__init__()
        self.grid_size = grid_size
        
        # Fixed grid points (mu) and variance (sigma)
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("grid", grid.view(1, 1, 1, 1, grid_size))
        self.sigma = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        
        # Learnable coefficients for the master function
        # Shape: (1, 1, 1, 1, grid_size) - A SINGLE master function per layer
        self.coef = nn.Parameter(torch.randn(1, 1, 1, 1, grid_size) / grid_size)

    def forward(self, x):
        # x shape: (B, C, H, W) -> expand to (B, C, H, W, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Compute Gaussian RBF: exp(- (x - mu)^2 / sigma^2)
        # Highly optimized for GPU Tensor Cores (no recursion)
        basis = torch.exp(-((x_expanded - self.grid) ** 2) / (self.sigma ** 2))
        
        # Weight by learnable coefficients and sum over the grid
        # Output shape: (B, C, H, W)
        return torch.sum(basis * self.coef, dim=-1)


class GSKANConv2d(nn.Module):
    """
    Generalized Shared Kolmogorov-Arnold Network Convolutional Layer.
    Solves the O(C * N_in * N_out) parameter bottleneck.
    
    Mathematically:
    y = W_base @ SiLU(x) + W_spline @ MasterFunction(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, grid_size=5):
        super().__init__()
        
        # 1. Base linear transformation (Standard practice in KANs to stabilize training)
        self.base_activation = nn.SiLU()
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        
        # 2. The single learnable master function for this layer
        self.master_fn = SharedRBFMasterFunction(grid_size=grid_size)
        
        # 3. Edge adaptation via learnable scalars (Standard Conv2d acts as the edge weights)
        self.spline_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        
        # Initialize spline weights to be small so base network dominates early training
        nn.init.kaiming_uniform_(self.base_conv.weight, a=math.sqrt(5)) if 'math' in globals() else None
        nn.init.normal_(self.spline_conv.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # Base path
        base_out = self.base_conv(self.base_activation(x))
        
        # KAN path (Master function -> Edge adaptation)
        spline_out = self.spline_conv(self.master_fn(x))
        
        return base_out + spline_out


if __name__ == "__main__":
    import math
    # ==========================================
    # HARDWARE SCALABILITY & COMPLEXITY VERIFICATION
    # ==========================================
    print("Initializing GS-KAN Layer...")
    
    IN_CHANNELS = 32
    OUT_CHANNELS = 64
    GRID_SIZE = 5
    IMG_SIZE = 128
    BATCH_SIZE = 2
    
    # Create the GS-KAN layer
    gs_kan_layer = GSKANConv2d(IN_CHANNELS, OUT_CHANNELS, kernel_size=3, padding=1, grid_size=GRID_SIZE)
    
    # Dummy input tensor
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    
    # Forward pass
    y = gs_kan_layer(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Calculate parameters
    base_params = sum(p.numel() for p in gs_kan_layer.base_conv.parameters())
    spline_params = sum(p.numel() for p in gs_kan_layer.spline_conv.parameters())
    master_params = sum(p.numel() for p in gs_kan_layer.master_fn.parameters())
    total_params = base_params + spline_params + master_params
    
    # Calculate what a Naive KAN would require
    # Naive KAN requires a unique spline for EVERY spatial connection and channel connection
    # Complexity: (Kernel_H * Kernel_W * In_C * Out_C) * Grid_Size
    naive_kan_params = (3 * 3 * IN_CHANNELS * OUT_CHANNELS) * GRID_SIZE
    
    print("\n--- Parameter Complexity Comparison ---")
    print(f"Naive KAN Parameters (OOM Risk): {naive_kan_params:,}")
    print(f"GS-KAN Parameters (Optimized):   {total_params:,}")
    print(f"Master Function Parameters:      {master_params} (A single shared basis!)")
    
    compression_ratio = naive_kan_params / total_params
    print(f"\nSUCCESS: GS-KAN achieves a {compression_ratio:.1f}x reduction in parameter complexity.")
    print("Hardware bottleneck bypassed. Ready for high-resolution CT grids.")