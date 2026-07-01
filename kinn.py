import torch
import torch.nn as nn
import torch.nn.functional as F

class MemristiveKirchhoffCell(nn.Module):
    """ 
    Spatial Domain: Memristive RC Circuit.
    Conductance adapts pixel-by-pixel based on the voltage state history.
    Preserves sharp edges (tumors) while rapidly decaying noise.
    """
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.C = nn.Parameter(torch.ones(1, channels, 1, 1))
        
        # Memristor Parameters
        self.G_base = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.G_mem = nn.Parameter(torch.zeros(1, channels, 1, 1)) # Learns how much to adapt
        
        self.current_extractor = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.current_extractor.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, v_prev):
        C_safe = F.softplus(self.C) + 1e-6
        
        # 1. Calculate Pixel-Adaptive Memductance M(V)
        # tanh(v_prev) bounds the state history between -1 and 1.
        M_t = F.softplus(self.G_base + self.G_mem * torch.tanh(v_prev)) + 1e-6
        
        # 2. Extract driving current (GELU is safe here in the spatial domain)
        I_t = F.gelu(self.current_extractor(x))
        
        # 3. Pixel-Adaptive ZOH Discretization
        decay = torch.exp(-M_t / C_safe)
        accumulation = I_t * (1.0 - decay) / M_t
        
        return (v_prev * decay) + accumulation

class LinearKirchhoffCell(nn.Module):
    """ Spectral Domain: STRICTLY LINEAR RC Circuit. Preserves wave phase. """
    def __init__(self, channels, kernel_size=1, padding=0):
        super().__init__()
        self.C = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.G = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.current_extractor = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)

    def forward(self, x, v_prev):
        C_safe = F.softplus(self.C) + 1e-6
        G_safe = F.softplus(self.G) + 1e-6
        
        I_t = self.current_extractor(x)
        
        decay = torch.exp(-G_safe / C_safe)
        accumulation = I_t * (1.0 - decay) / G_safe
        return (v_prev * decay) + accumulation

class SpectralKirchhoffCell2d(nn.Module):
    def __init__(self, channels, modes1=16, modes2=16):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.knc_real = LinearKirchhoffCell(channels)
        self.knc_imag = LinearKirchhoffCell(channels)

    def forward(self, x, v_prev):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        v_prev_ft = torch.fft.rfft2(v_prev)
        
        out_ft = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.complex64, device=x.device)
        
        real_next = self.knc_real(x_ft[:, :, :self.modes1, :self.modes2].real, v_prev_ft[:, :, :self.modes1, :self.modes2].real)
        imag_next = self.knc_imag(x_ft[:, :, :self.modes1, :self.modes2].imag, v_prev_ft[:, :, :self.modes1, :self.modes2].imag)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.complex(real_next, imag_next)
        return torch.fft.irfft2(out_ft, s=(H, W))

class KINN_Block(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.spectral_knc = SpectralKirchhoffCell2d(channels)
        self.spatial_knc = MemristiveKirchhoffCell(channels) # Upgraded to Memristor
        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x, v_state):
        v_state = self.spectral_knc(x, v_state)
        v_state = self.spatial_knc(x, v_state)
        return self.norm(v_state)