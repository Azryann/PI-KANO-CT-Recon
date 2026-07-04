import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_dfst_diagram():
    # 1. Create a dummy 2D frequency spectrum (Gaussian drop-off)
    N = 512
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Simulate CT frequency spectrum (high energy at center, drops off)
    spectrum = np.exp(-10 * R) + 0.1 * np.exp(-50 * R**2)
    spectrum = np.log1p(spectrum * 1000) # Log scale for visibility
    
    plt.figure(figsize=(8, 8))
    plt.imshow(spectrum, cmap='magma', extent=[-1, 1, -1, 1])
    
    # 2. Draw Polar Sampling Lines (The Fourier Slices)
    num_lines = 18
    angles = np.linspace(0, np.pi, num_lines, endpoint=False)
    for theta in angles:
        x_line = [-np.cos(theta), np.cos(theta)]
        y_line = [-np.sin(theta), np.sin(theta)]
        plt.plot(x_line, y_line, color='cyan', alpha=0.6, linewidth=1.5)
        
    # 3. Highlight the "Gridding Artifact Zones" (Sparse High-Frequency Corners)
    rects = [
        patches.Rectangle((0.5, 0.5), 0.5, 0.5, linewidth=2, edgecolor='red', facecolor='red', alpha=0.2),
        patches.Rectangle((-1.0, 0.5), 0.5, 0.5, linewidth=2, edgecolor='red', facecolor='red', alpha=0.2),
        patches.Rectangle((0.5, -1.0), 0.5, 0.5, linewidth=2, edgecolor='red', facecolor='red', alpha=0.2),
        patches.Rectangle((-1.0, -1.0), 0.5, 0.5, linewidth=2, edgecolor='red', facecolor='red', alpha=0.2)
    ]
    for rect in rects:
        plt.gca().add_patch(rect)
        
    plt.text(0.75, 0.75, "Sparse\nSampling\n(Artifacts)", color='white', fontsize=12, 
             fontweight='bold', ha='center', va='center')
    
    plt.title("Differentiable Fourier-Slice Theorem (DFST) Sampling", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Frequency $k_x$", fontsize=14)
    plt.ylabel("Frequency $k_y$", fontsize=14)
    
    # Draw a circle to show the Nyquist limit of the polar sampling
    circle = patches.Circle((0, 0), 1.0, linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
    plt.gca().add_patch(circle)
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig("fig_dfst_concept.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'fig_dfst_concept.png'")

if __name__ == "__main__":
    generate_dfst_diagram()