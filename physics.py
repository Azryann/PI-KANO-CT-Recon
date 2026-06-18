import torch
import torch.nn as nn

class ExactDiscreteRadonFn(torch.autograd.Function):
    """
    Custom PyTorch Autograd Function for the Exact Discrete Radon Transform.
    Forward pass: y = A @ x
    Backward pass (Exact Adjoint): dx = A^T @ dy
    This strictly prevents 'adjoint mismatch' in proximal gradient schemes.
    """
    @staticmethod
    def forward(ctx, x, system_matrix, img_shape, sino_shape):
        # x shape: (Batch, Channels, H, W)
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions and transpose for sparse multiplication
        # x_flat shape: (C*H*W, B)
        x_flat = x.view(B, -1).t()
        
        # Forward projection: A @ x
        # system_matrix shape: (M, N) where M = angles*detectors, N = C*H*W
        y_flat = torch.sparse.mm(system_matrix, x_flat)
        
        # Reshape back to sinogram dimensions: (Batch, Channels, Angles, Detectors)
        y = y_flat.t().view(B, C, sino_shape[0], sino_shape[1])
        
        # Save for backward pass
        ctx.save_for_backward(system_matrix)
        ctx.img_shape = img_shape
        ctx.B = B
        ctx.C = C
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        system_matrix, = ctx.saved_tensors
        B, C = ctx.B, ctx.C
        H, W = ctx.img_shape
        
        # Flatten and transpose grad_output: (M, B)
        grad_y_flat = grad_output.view(B, -1).t()
        
        # Exact Discrete Adjoint: A^T @ dy
        grad_x_flat = torch.sparse.mm(system_matrix.t(), grad_y_flat)
        
        # Reshape back to image dimensions: (Batch, Channels, H, W)
        grad_input = grad_x_flat.t().view(B, C, H, W)
        
        # Return gradients for inputs (None for non-tensor arguments)
        return grad_input, None, None, None


class RadonPhysics(nn.Module):
    def __init__(self, img_size, num_angles, num_detectors, device='cuda'):
        super().__init__()
        self.img_shape = (img_size, img_size)
        self.sino_shape = (num_angles, num_detectors)
        self.device = device
        
        # Generate a dummy sparse system matrix for prototyping.
        # In production (Kaggle), this will be replaced by the LoDoPaB-CT specific geometry matrix.
        self.system_matrix = self._generate_dummy_system_matrix().to(self.device)

    def _generate_dummy_system_matrix(self):
        """
        Generates a highly sparse random matrix to simulate ray-driven CT physics.
        Memory efficient for 4GB VRAM.
        """
        N = self.img_shape[0] * self.img_shape[1]
        M = self.sino_shape[0] * self.sino_shape[1]
        
        # Simulate ~5% sparsity (rays only hit a fraction of pixels)
        nnz = int(M * N * 0.05)
        
        indices = torch.randint(0, M, (1, nnz))
        indices = torch.cat([indices, torch.randint(0, N, (1, nnz))], dim=0)
        values = torch.rand(nnz)
        
        # Create sparse tensor
        A = torch.sparse_coo_tensor(indices, values, (M, N)).coalesce()
        return A

    def forward(self, x):
        """ Forward Radon Transform """
        return ExactDiscreteRadonFn.apply(x, self.system_matrix, self.img_shape, self.sino_shape)

    def adjoint(self, y):
        """ Explicit Exact Discrete Adjoint (Backprojection) """
        B, C, _, _ = y.shape
        y_flat = y.view(B, -1).t()
        x_flat = torch.sparse.mm(self.system_matrix.t(), y_flat)
        return x_flat.t().view(B, C, self.img_shape[0], self.img_shape[1])


if __name__ == "__main__":
    # ==========================================
    # MATHEMATICAL VERIFICATION OF EXACT ADJOINT
    # ==========================================
    print("Initializing Physics Operator...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Small dimensions for local RTX 3050 prototyping
    IMG_SIZE = 64
    ANGLES = 30
    DETECTORS = 64
    BATCH_SIZE = 2
    CHANNELS = 1
    
    physics = RadonPhysics(IMG_SIZE, ANGLES, DETECTORS, device=device)
    
    # 1. Create random image (x) and random sinogram (y)
    x = torch.randn(BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randn(BATCH_SIZE, CHANNELS, ANGLES, DETECTORS, device=device)
    
    # 2. Compute Forward (Ax) and Adjoint (A^T y)
    Ax = physics.forward(x)
    ATy = physics.adjoint(y)
    
    # 3. Compute inner products: <Ax, y> and <x, A^T y>
    # Flatten tensors to compute standard dot product
    inner_product_1 = torch.sum(Ax * y)
    inner_product_2 = torch.sum(x * ATy)
    
    print(f"\n--- Adjoint Mismatch Verification ---")
    print(f"<Ax, y>   = {inner_product_1.item():.6f}")
    print(f"<x, A^T y> = {inner_product_2.item():.6f}")
    
    # 4. Verify mathematical equality
    difference = torch.abs(inner_product_1 - inner_product_2).item()
    print(f"Absolute Difference: {difference:.6e}")
    
    if difference < 1e-4:
        print("SUCCESS: Exact Discrete Adjoint mathematically verified. Monotone inclusion properties preserved.")
    else:
        print("ERROR: Adjoint mismatch detected!")