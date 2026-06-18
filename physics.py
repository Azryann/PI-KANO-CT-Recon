import os
import torch
import torch.nn as nn

class ExactDiscreteRadonFn(torch.autograd.Function):
    """
    Custom PyTorch Autograd Function for the Exact Discrete Radon Transform.
    Uses highly optimized CSR sparse matrices to prevent OOM.
    """
    @staticmethod
    def forward(ctx, x, system_matrix_csr, system_matrix_t_csr, img_shape, sino_shape):
        B, C, H, W = x.shape
        x_flat = x.view(B, -1).t()
        
        # Forward projection: A @ x
        y_flat = torch.sparse.mm(system_matrix_csr, x_flat)
        y = y_flat.t().view(B, C, sino_shape[0], sino_shape[1])
        
        # Save the pre-computed transpose for the backward pass
        ctx.save_for_backward(system_matrix_t_csr)
        ctx.img_shape = img_shape
        ctx.B = B
        ctx.C = C
        return y

    @staticmethod
    def backward(ctx, grad_output):
        system_matrix_t_csr, = ctx.saved_tensors
        B, C = ctx.B, ctx.C
        H, W = ctx.img_shape
        
        grad_y_flat = grad_output.view(B, -1).t()
        
        # Exact Discrete Adjoint: A^T @ dy
        grad_x_flat = torch.sparse.mm(system_matrix_t_csr, grad_y_flat)
        grad_input = grad_x_flat.t().view(B, C, H, W)
        
        return grad_input, None, None, None, None


class RadonPhysics(nn.Module):
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, device='cuda'):
        super().__init__()
        self.img_shape = (img_size, img_size)
        self.sino_shape = (num_angles, num_detectors)
        self.device = device
        
        matrix_path = "lodopab_geometry.pt"
        
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Geometry matrix not found at {matrix_path}.")
            
        print("Loading Exact LoDoPaB Geometry Matrix into VRAM...")
        coo_matrix = torch.load(matrix_path).to(self.device)
        
        print("Optimizing matrix format for GPU (Converting COO to CSR)...")
        # CSR format is strictly required for memory-efficient sparse.mm
        self.system_matrix_csr = coo_matrix.to_sparse_csr()
        
        # Pre-compute the transpose and convert to CSR to prevent OOM during training
        self.system_matrix_t_csr = coo_matrix.t().to_sparse_csr()
        
        # Delete the original COO matrix and clear cache to free up ~3.3GB of VRAM
        del coo_matrix
        torch.cuda.empty_cache()
        
        print("Physics Operator Ready.")

    def forward(self, x):
        """ Forward Radon Transform """
        return ExactDiscreteRadonFn.apply(
            x, self.system_matrix_csr, self.system_matrix_t_csr, self.img_shape, self.sino_shape
        )

    def adjoint(self, y):
        """ Explicit Exact Discrete Adjoint (Backprojection) """
        B, C, _, _ = y.shape
        y_flat = y.view(B, -1).t()
        x_flat = torch.sparse.mm(self.system_matrix_t_csr, y_flat)
        return x_flat.t().view(B, C, self.img_shape[0], self.img_shape[1])