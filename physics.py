import torch
import torch.nn as nn
import numpy as np
import astra

class AstraRadonFn(torch.autograd.Function):
    """
    Matrix-Free PyTorch Autograd Function.
    Uses ASTRA's explicit FP_CUDA and BP_CUDA algorithms to compute the exact 
    forward and adjoint transforms on-the-fly, consuming ZERO memory for the matrix.
    """
    @staticmethod
    def forward(ctx, x, vol_geom, proj_geom):
        B, C, H, W = x.shape
        device = x.device
        
        # Move to CPU/Numpy for ASTRA C++ interface
        x_np = x.detach().cpu().numpy()
        
        num_angles = proj_geom['ProjectionAngles'].shape[0]
        num_detectors = proj_geom['DetectorCount']
        y_np = np.zeros((B, C, num_angles, num_detectors), dtype=np.float32)
        
        # Compute Forward Projection on GPU using FP_CUDA
        for b in range(B):
            for c in range(C):
                vol_id = astra.data2d.create('-vol', vol_geom, x_np[b, c])
                sino_id = astra.data2d.create('-sino', proj_geom)
                
                cfg = astra.astra_dict('FP_CUDA')
                cfg['VolumeDataId'] = vol_id
                cfg['ProjectionDataId'] = sino_id
                
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
                
                y_np[b, c] = astra.data2d.get(sino_id)
                
                # Cleanup C++ memory
                astra.algorithm.delete(alg_id)
                astra.data2d.delete(vol_id)
                astra.data2d.delete(sino_id)
                
        ctx.vol_geom = vol_geom
        ctx.proj_geom = proj_geom
        
        return torch.from_numpy(y_np).to(device)

    @staticmethod
    def backward(ctx, grad_output):
        vol_geom = ctx.vol_geom
        proj_geom = ctx.proj_geom
        
        B, C, Angles, Detectors = grad_output.shape
        device = grad_output.device
        
        grad_np = grad_output.detach().cpu().numpy()
        
        H = vol_geom['GridRowCount']
        W = vol_geom['GridColCount']
        grad_x_np = np.zeros((B, C, H, W), dtype=np.float32)
        
        # Compute Exact Adjoint (Backprojection) on GPU using BP_CUDA
        for b in range(B):
            for c in range(C):
                sino_id = astra.data2d.create('-sino', proj_geom, grad_np[b, c])
                vol_id = astra.data2d.create('-vol', vol_geom)
                
                cfg = astra.astra_dict('BP_CUDA')
                # FIX: ASTRA requires 'ReconstructionDataId' for Backprojection
                cfg['ReconstructionDataId'] = vol_id  
                cfg['ProjectionDataId'] = sino_id
                
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
                
                grad_x_np[b, c] = astra.data2d.get(vol_id)
                
                # Cleanup C++ memory
                astra.algorithm.delete(alg_id)
                astra.data2d.delete(vol_id)
                astra.data2d.delete(sino_id)
                
        return torch.from_numpy(grad_x_np).to(device), None, None


class RadonPhysics(nn.Module):
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, device='cuda'):
        super().__init__()
        self.device = device
        
        # Define ASTRA Geometries (Lightweight dictionaries, 0 VRAM!)
        self.vol_geom = astra.create_vol_geom(img_size, img_size)
        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, num_detectors, angles)
        
        print("ASTRA Matrix-Free CUDA Operator Initialized (0 VRAM footprint).")

    def forward(self, x):
        """ Forward Radon Transform """
        return AstraRadonFn.apply(x, self.vol_geom, self.proj_geom)

    def adjoint(self, y):
        """ Explicit Exact Discrete Adjoint (Backprojection) """
        B, C, Angles, Detectors = y.shape
        y_np = y.detach().cpu().numpy()
        
        H = self.vol_geom['GridRowCount']
        W = self.vol_geom['GridColCount']
        x_np = np.zeros((B, C, H, W), dtype=np.float32)
        
        for b in range(B):
            for c in range(C):
                sino_id = astra.data2d.create('-sino', self.proj_geom, y_np[b, c])
                vol_id = astra.data2d.create('-vol', self.vol_geom)
                
                cfg = astra.astra_dict('BP_CUDA')
                # FIX: ASTRA requires 'ReconstructionDataId' for Backprojection
                cfg['ReconstructionDataId'] = vol_id  
                cfg['ProjectionDataId'] = sino_id
                
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
                
                x_np[b, c] = astra.data2d.get(vol_id)
                
                astra.algorithm.delete(alg_id)
                astra.data2d.delete(vol_id)
                astra.data2d.delete(sino_id)
                
        return torch.from_numpy(x_np).to(self.device)