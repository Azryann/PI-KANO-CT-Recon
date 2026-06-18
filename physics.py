import os
import torch
import torch.nn as nn

class ExactDiscreteRadonFn(torch.autograd.Function):
    """
    Custom PyTorch Autograd Function for the Exact Discrete Radon Transform.
    Forward pass: y = A @ x
    Backward pass (Exact Adjoint): dx = A^T @ dy
    """
    @staticmethod
    def forward(ctx, x, system_matrix, img_shape, sino_shape):
        B, C, H, W = x.shape
        x_flat = x.view(B, -1).t()
        
        # Forward projection: A @ x
        y_flat = torch.sparse.mm(system_matrix, x_flat)
        y = y_flat.t().view(B, C, sino_shape[0], sino_shape[1])
        
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
        
        grad_y_flat = grad_output.view(B, -1).t()
        
        # Exact Discrete Adjoint: A^T @ dy
        grad_x_flat = torch.sparse.mm(system_matrix.t(), grad_y_flat)
        grad_input = grad_x_flat.t().view(B, C, H, W)
        
        return grad_input, None, None, None


class RadonPhysics(nn.Module):
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, device='cuda'):
        super().__init__()
        self.img_shape = (img_size, img_size)
        self.sino_shape = (num_angles, num_detectors)
        self.device = device
        
        matrix_path = "lodopab_geometry.pt"
        
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(
                f"Geometry matrix not found at {matrix_path}. "
                "Please run the ASTRA matrix generation script first."
            )
            
        print("Loading Exact LoDoPaB Geometry Matrix into VRAM...")
        # Load the exact matrix and move it to the GPU
        self.system_matrix = torch.load(matrix_path).to(self.device)
        print("Physics Operator Ready.")

    def forward(self, x):
        """ Forward Radon Transform """
        return ExactDiscreteRadonFn.apply(x, self.system_matrix, self.img_shape, self.sino_shape)

    def adjoint(self, y):
        """ Explicit Exact Discrete Adjoint (Backprojection) """
        B, C, _, _ = y.shape
        y_flat = y.view(B, -1).t()
        x_flat = torch.sparse.mm(self.system_matrix.t(), y_flat)
        return x_flat.t().view(B, C, self.img_shape[0], self.img_shape[1])