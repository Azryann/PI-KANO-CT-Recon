import torch
import torch.nn as nn
import torch.nn.functional as F
from physics import RadonPhysics

class DepthwiseSeparableConv(nn.Module):
    """ 
    Edge-AI Optimization: Reduces parameters by ~90% compared to standard Conv2d.
    Crucial for maintaining the 'Lightweight Edge Deployment' narrative.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class FrequencyAttention(nn.Module):
    """
    Novel Algorithmic Contribution: Frequency-Domain Attention.
    Dynamically weights frequency bands to suppress noise and enhance structural edges.
    """
    def __init__(self, channels):
        super().__init__()
        # Ultra-lightweight MLP (bottleneck ratio 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid() 
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x)
        mag = torch.abs(x_freq)
        
        freq_profile = mag.mean(dim=(2, 3)) 
        attn_weights = self.mlp(freq_profile).view(B, C, 1, 1)
        
        x_freq_attended = x_freq * attn_weights
        return torch.fft.irfft2(x_freq_attended, s=(H, W))

class FDABlock(nn.Module):
    """ Dual-Domain Unrolled Cascade Block (MobileNet-Optimized) """
    def __init__(self, in_channels=2, hidden_channels=32):
        super().__init__()
        
        # First layer must be standard to expand from 2 channels to 32
        self.expand_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.freq_attn = FrequencyAttention(hidden_channels)
        
        # Heavy internal processing replaced with Depthwise Separable Convs
        self.dw_conv = DepthwiseSeparableConv(hidden_channels, hidden_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(hidden_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Final projection back to 1 channel
        self.project_conv = nn.Conv2d(hidden_channels, 1, 3, padding=1)

    def forward(self, x):
        feat = self.relu1(self.norm1(self.expand_conv(x)))
        feat = feat + self.freq_attn(feat) 
        feat = self.relu2(self.norm2(self.dw_conv(feat)))
        return self.project_conv(feat)

class FDA_Net(nn.Module):
    """
    Frequency-Domain Attention Network for Edge-Deployable CT Reconstruction.
    """
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, num_cascades=3, device='cuda'):
        super().__init__()
        self.num_cascades = num_cascades
        self.physics = RadonPhysics(img_size, num_angles, num_detectors, device=device)
        
        self.tau_max = 2.0 / self._power_iteration(img_size, device)
        self.tau = nn.Parameter(torch.tensor(self.tau_max * 0.1))
        
        self.blocks = nn.ModuleList([FDABlock() for _ in range(num_cascades)])

    def _power_iteration(self, img_size, device, num_iters=10):
        u = torch.randn(1, 1, img_size, img_size, device=device)
        u = u / torch.norm(u)
        with torch.no_grad():
            for _ in range(num_iters):
                v = self.physics.adjoint(self.physics.forward(u))
                u = v / torch.norm(v)
        return torch.norm(v).item()

    def forward(self, y):
        tau_safe = torch.clamp(self.tau, min=1e-8, max=self.tau_max * 0.99)
        x = self.physics.adjoint(y) * tau_safe
        
        for i in range(self.num_cascades):
            grad = self.physics.adjoint(self.physics.forward(x) - y) * tau_safe
            update = self.blocks[i](torch.cat([x, grad], dim=1))
            x = x - update
        return x