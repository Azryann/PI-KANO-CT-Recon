import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

def generate_dfst_diagram():
    # Use LaTeX-style fonts without requiring a full LaTeX installation
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
    })
    
    N = 512
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Base CT Spectrum
    spectrum = np.exp(-10 * R) + 0.1 * np.exp(-50 * R**2)
    spectrum = np.log1p(spectrum * 1000)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # ==========================================
    # PANEL 1: The Problem (Sparse Sampling)
    # ==========================================
    ax1 = axes[0]
    ax1.imshow(spectrum, cmap='magma', extent=[-1, 1, -1, 1])
    
    # Polar Lines
    angles = np.linspace(0, np.pi, 18, endpoint=False)
    for theta in angles:
        ax1.plot([-np.cos(theta), np.cos(theta)], [-np.sin(theta), np.sin(theta)], color='cyan', alpha=0.5, lw=1.5)
        
    # Soft Shading for Artifact Zones (Using a radial gradient mask)
    corner_mask = np.clip((R - 0.7) * 2, 0, 1) * (np.abs(X) > 0.3) * (np.abs(Y) > 0.3)
    ax1.imshow(corner_mask, cmap='Reds', extent=[-1, 1, -1, 1], alpha=0.4)
    
    ax1.text(0.75, 0.75, "Sparse\nSampling\n(Artifacts)", color='white', fontsize=12, fontweight='bold', ha='center', va='center')
    ax1.add_patch(patches.Circle((0, 0), 1.0, lw=2, edgecolor='white', facecolor='none', ls='--'))
    ax1.set_title("A. DFST Polar Sampling", fontweight='bold', pad=15)
    ax1.set_xlabel("Frequency $k_x$")
    ax1.set_ylabel("Frequency $k_y$")
    
    # ==========================================
    # PANEL 2: The Solution (FDA Attention Map)
    # ==========================================
    ax2 = axes[1]
    ax2.imshow(spectrum, cmap='magma', extent=[-1, 1, -1, 1])
    
    # Simulate the learned FDA Attention Map (Focuses on high-freq corners)
    attention_map = np.clip((R - 0.4), 0, 1)**2
    # Overlay the attention heatmap (Hot colormap)
    im = ax2.imshow(attention_map, cmap='inferno', extent=[-1, 1, -1, 1], alpha=0.6)
    
    ax2.text(0.75, 0.75, "High Attention\n(Artifact Suppression)", color='white', fontsize=12, fontweight='bold', ha='center', va='center')
    ax2.add_patch(patches.Circle((0, 0), 1.0, lw=2, edgecolor='white', facecolor='none', ls='--'))
    ax2.set_title("B. Learned Frequency-Domain Attention", fontweight='bold', pad=15)
    ax2.set_xlabel("Frequency $k_x$")
    
    plt.tight_layout()
    plt.savefig("fig_dfst_concept.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'fig_dfst_concept.png'")

if __name__ == "__main__":
    generate_dfst_diagram()