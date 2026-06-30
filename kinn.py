import torch
import torch.nn as nn
import torch.nn.functional as F

class KirchhoffNeuralCell2d(nn.Module):
    """
    Second-Order RLC Kirchhoff Neural Cell.
    Adds an Inductor (L) to provide physical momentum, allowing the network 
    to accelerate past local minima and beat heavy CNN baselines.
    """
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        # Learnable Physical Parameters
        self.C = nn.Parameter(torch.ones(1, channels, 1, 1)) # Capacitance
        self.G = nn.Parameter(torch.ones(1, channels, 1, 1)) # Conductance
        self.L = nn.Parameter(torch.ones(1, channels, 1, 1)) # Inductance (Momentum)
        
        self.current_extractor = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.current_extractor.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, v_prev, v_momentum):
        C_safe = F.softplus(self.C) + 1e-6
        G_safe = F.softplus(self.G) + 1e-6
        L_safe = F.softplus(self.L) + 1e-6
        
        # The "Diode" (Non-linear activation)
        I_t = F.gelu(self.current_extractor(x))
        
        # RLC Discrete Momentum Update
        # Momentum decay is governed by the Inductor (L) and Resistor (G)
        momentum_decay = torch.exp(-G_safe / L_safe)
        
        # Voltage decay is governed by the Capacitor (C)
        voltage_decay = torch.exp(-G_safe / C_safe)
        
        # Update Momentum (Current flowing through the inductor)
        v_momentum_next = (v_momentum * momentum_decay) + I_t
        
        # Update Voltage (Charge accumulating in the capacitor, driven by momentum)
        accumulation = v_momentum_next * (1.0 - voltage_decay) / G_safe
        v_next = (v_prev * voltage_decay) + accumulation
        
        return v_next, v_momentum_next


class SpectralKirchhoffCell2d(nn.Module):
    def __init__(self, channels, modes1=16, modes2=16):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.knc_real = KirchhoffNeuralCell2d(channels, kernel_size=1, padding=0)
        self.knc_imag = KirchhoffNeuralCell2d(channels, kernel_size=1, padding=0)

    def forward(self, x, v_prev, v_momentum):
        B, C, H, W = x.shape
        
        x_ft = torch.fft.rfft2(x)
        v_prev_ft = torch.fft.rfft2(v_prev)
        v_mom_ft = torch.fft.rfft2(v_momentum)
        
        out_ft = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.complex64, device=x.device)
        out_mom_ft = torch.zeros_like(out_ft)
        
        real_next, real_mom_next = self.knc_real(
            x_ft[:, :, :self.modes1, :self.modes2].real, 
            v_prev_ft[:, :, :self.modes1, :self.modes2].real,
            v_mom_ft[:, :, :self.modes1, :self.modes2].real
        )
        imag_next, imag_mom_next = self.knc_imag(
            x_ft[:, :, :self.modes1, :self.modes2].imag, 
            v_prev_ft[:, :, :self.modes1, :self.modes2].imag,
            v_mom_ft[:, :, :self.modes1, :self.modes2].imag
        )
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.complex(real_next, imag_next)
        out_mom_ft[:, :, :self.modes1, :self.modes2] = torch.complex(real_mom_next, imag_mom_next)
        
        return torch.fft.irfft2(out_ft, s=(H, W)), torch.fft.irfft2(out_mom_ft, s=(H, W))


class KINN_Block(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.spectral_knc = SpectralKirchhoffCell2d(channels)
        self.spatial_knc = KirchhoffNeuralCell2d(channels)
        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x, v_state, v_momentum):
        v_state, v_momentum = self.spectral_knc(x, v_state, v_momentum)
        v_state, v_momentum = self.spatial_knc(x, v_state, v_momentum)
        return self.norm(v_state), v_momentum