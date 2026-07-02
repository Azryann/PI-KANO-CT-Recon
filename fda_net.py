import torch
import torch.nn as nn
import torch.nn.functional as F
from physics import RadonPhysics

class FrequencyAttention(nn.Module):
    """
    Novel Algorithmic Contribution: Frequency-Domain Attention.
    Dynamically weights frequency bands to suppress noise and enhance structural edges.
    """
    def __init__(self, channels):
        super().__init__()
        # Lightweight MLP to learn frequency weights
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid() # Outputs attention weights between 0 and 1
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Transform to Frequency Domain
        x_freq = torch.fft.rfft2(x)
        
        # 2. Extract Magnitude (Amplitude of frequencies)
        mag = torch.abs(x_freq)
        
        # 3. Global Average Pooling over spatial frequencies to get channel-wise frequency profile
        freq_profile = mag.mean(dim=(2, 3)) # Shape: (B, C)
        
        # 4. Learn Attention Weights
        attn_weights = self.mlp(freq_profile).view(B, C, 1, 1)
        
        # 5. Apply Attention in Frequency Domain
        x_freq_attended = x_freq * attn_weights
        
        # 6. Return to Spatial Domain
        return torch.fft.irfft2(x_freq_attended, s=(H, W))

class FDABlock(nn.Module):
    """ Dual-Domain Unrolled Cascade Block """
    def __init__(self, in_channels=2, hidden_channels=32):
        super().__init__()
        self.spatial_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # The Novel Frequency Attention Module
        self.freq_attn = FrequencyAttention(hidden_channels)
        
        self.spatial_conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1)
        )

    def forward(self, x):
        feat = self.spatial_conv1(x)
        feat = feat + self.freq_attn(feat) # Residual Frequency Attention
        return self.spatial_conv2(feat)

class FDA_Net(nn.Module):
    """
    Frequency-Domain Attention Network for Edge-Deployable CT Reconstruction.
    """
    def __init__(self, img_size=362, num_angles=1000, num_detectors=513, num_cascades=3, device='cuda'):
        super().__init__()
        self.num_cascades = num_cascades
        self.physics = RadonPhysics(img_size, num_angles, num_detectors, device=device)
        
        # Power Iteration for strict proximal bounds
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