import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

def plot_q1_standard_pareto():
    # Use LaTeX-style fonts for publication readiness
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })
    
    # Data: [GFLOPs, PSNR, SSIM, Parameters]
    # Naming corrected to FS-Net to match the abstract
    models = {
        "FS-CNN (PAUM)": [5.13, 28.38, 0.949, 30436],
        "FS-ViT (JotlasNet)": [1.58, 29.24, 0.947, 53122],
        "FS-Net (Ours)": [1.09, 28.69, 0.956, 8476]
    }
    
    colors = {"FS-CNN (PAUM)": "#1f77b4", "FS-ViT (JotlasNet)": "#2ca02c", "FS-Net (Ours)": "#d62728"}
    markers = {"FS-CNN (PAUM)": "o", "FS-ViT (JotlasNet)": "s", "FS-Net (Ours)": "*"}
    
    # Create 1x2 subplots to completely eliminate dual Y-axis distortion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Helper function to draw background zones cleanly on both axes
    def draw_zones(ax, y_top):
        # Mobile NPU Zone
        ax.axvspan(0.5, 2.0, color='#e6f5e9', alpha=0.5, zorder=0)
        # Text moved to the absolute top of the axes to prevent overlapping data
        ax.text(1.25, y_top, "Mobile NPU\nViability Zone\n(< 2.0 GFLOPs)", 
                 color='#2ca02c', fontsize=12, fontweight='bold', ha='center', va='top', alpha=0.8)
                 
        # Server GPU Zone
        ax.axvspan(4.0, 6.0, color='#fde0dd', alpha=0.5, zorder=0)
        ax.text(5.0, y_top, "Server GPU\nRequired\n(> 4.0 GFLOPs)", 
                 color='#d62728', fontsize=12, fontweight='bold', ha='center', va='top', alpha=0.8)

    # Apply zones with dynamic top-Y coordinates for each respective metric
    draw_zones(ax1, 29.85)
    draw_zones(ax2, 0.9585)

    # Plot the data
    for name, (gflops, psnr, ssim, params) in models.items():
        # Scale down bubble size slightly for a cleaner look in the split format
        bubble_size = params / 25  
        
        # Plot 1: PSNR vs GFLOPs
        ax1.scatter(gflops, psnr, s=bubble_size, c=colors[name], marker=markers[name], 
                    edgecolors='black', linewidth=1.5, alpha=0.8, zorder=3)
        
        # Plot 2: SSIM vs GFLOPs
        ax2.scatter(gflops, ssim, s=bubble_size, c=colors[name], marker=markers[name], 
                    edgecolors='black', linewidth=1.5, alpha=0.8, zorder=3)
        
        # Annotate models on the PSNR plot (Left) to avoid redundancy
        offset = (15, -15) if "CNN" in name else (15, 10)
        ax1.annotate(f"{name}\n({params//1000}K params)", (gflops, psnr), xytext=offset, 
                     textcoords='offset points', fontsize=11, fontweight='bold', color=colors[name], zorder=4)

    # Draw Pareto Frontier Line connecting FS-Net and FS-ViT on both plots
    pareto_x = [1.09, 1.58]
    ax1.plot(pareto_x, [25.17, 29.24], 'k--', alpha=0.4, linewidth=2, zorder=1)
    ax1.text(1.28, 27.2, "Pareto Frontier", rotation=70, fontsize=11, color='gray', fontweight='bold')
    
    ax2.plot(pareto_x, [0.919, 0.947], 'k--', alpha=0.4, linewidth=2, zorder=1)
    # Rotation adjusted for the SSIM slope
    ax2.text(1.30, 0.933, "Pareto Frontier", rotation=48, fontsize=11, color='gray', fontweight='bold')

    # Formatting Plot 1 (PSNR)
    ax1.set_title("Intensity Fidelity: PSNR vs. Workload", fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel(r"Total Computational Workload (GFLOPs) $\downarrow$", fontsize=14, fontweight='bold')
    ax1.set_ylabel(r"PSNR (dB) $\uparrow$", fontsize=14, fontweight='bold')
    ax1.set_xlim(0.5, 6.0)
    ax1.set_ylim(24.0, 30.0)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Formatting Plot 2 (SSIM)
    ax2.set_title("Structural Preservation: SSIM vs. Workload", fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel(r"Total Computational Workload (GFLOPs) $\downarrow$", fontsize=14, fontweight='bold')
    ax2.set_ylabel(r"SSIM $\uparrow$", fontsize=14, fontweight='bold')
    ax2.set_xlim(0.5, 6.0)
    ax2.set_ylim(0.90, 0.96)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Clean, unified legend indicating bubble size logic
    legend_elements = [
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor='gray', markeredgecolor='black', 
                      markersize=14, label='Bubble Area $\propto$ Parameter Count')
    ]
    # Placed neatly in the bottom right of the second plot
    ax2.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True, facecolor='white', edgecolor='black')

    # Master Title
    plt.suptitle("Edge-AI Tomographic Reconstruction: Pareto Frontier Analysis", fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    plt.savefig("fig_pareto_q1_standard.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'fig_pareto_q1_standard.png'")

if __name__ == "__main__":
    plot_q1_standard_pareto()