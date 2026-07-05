import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def plot_advanced_pareto():
    # Use LaTeX-style fonts
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
    models = {
        "FS-CNN (PAUM)": [5.13, 28.38, 0.949, 30436],
        "FS-ViT (JotlasNet)": [1.58, 29.24, 0.947, 53122],
        "FDA-Net (Ours)": [1.09, 25.17, 0.919, 8476]
    }
    
    colors = {"FS-CNN (PAUM)": "#1f77b4", "FS-ViT (JotlasNet)": "#2ca02c", "FDA-Net (Ours)": "#d62728"}
    markers = {"FS-CNN (PAUM)": "o", "FS-ViT (JotlasNet)": "s", "FDA-Net (Ours)": "*"}
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Create secondary Y-axis for SSIM
    ax2 = ax1.twinx()
    
    # Draw the "Edge-AI Viability Zone" (GFLOPs < 2.0)
    ax1.axvspan(0.5, 2.0, color='#e6f5e9', alpha=0.5, zorder=0)
    ax1.text(1.25, 29.5, "Mobile NPU\nViability Zone\n(< 2.0 GFLOPs)", 
             color='#2ca02c', fontsize=12, fontweight='bold', ha='center', va='top', alpha=0.7)
             
    # Draw the "Server Only Zone" (GFLOPs > 4.0)
    ax1.axvspan(4.0, 6.0, color='#fde0dd', alpha=0.5, zorder=0)
    ax1.text(5.0, 29.5, "Server GPU\nRequired\n(> 4.0 GFLOPs)", 
             color='#d62728', fontsize=12, fontweight='bold', ha='center', va='top', alpha=0.7)

    # Plot the data
    for name, (gflops, psnr, ssim, params) in models.items():
        # Bubble size proportional to parameter count
        bubble_size = params / 30  
        
        # Plot PSNR on primary axis (Solid bubbles)
        ax1.scatter(gflops, psnr, s=bubble_size, c=colors[name], marker=markers[name], 
                   edgecolors='black', linewidth=1.5, alpha=0.8, zorder=3)
        
        # Plot SSIM on secondary axis (Hollow markers with dashed edges)
        ax2.scatter(gflops, ssim, s=bubble_size*0.5, facecolors='none', edgecolors=colors[name], 
                   marker=markers[name], linewidth=2, linestyle='--', alpha=0.8, zorder=3)
        
        # Annotate the main bubbles
        offset = (15, -15) if "CNN" in name else (15, 10)
        ax1.annotate(f"{name}\n({params//1000}K params)", (gflops, psnr), xytext=offset, 
                    textcoords='offset points', fontsize=11, fontweight='bold', color=colors[name], zorder=4)

    # Draw Pareto Frontier Line (Connecting FDA-Net and FS-ViT)
    pareto_x = [1.09, 1.58]
    pareto_psnr = [25.17, 29.24]
    ax1.plot(pareto_x, pareto_psnr, 'k--', alpha=0.4, linewidth=2, zorder=1)
    ax1.text(1.33, 27.2, "Pareto Frontier", rotation=75, fontsize=11, color='gray', fontweight='bold')

    # Formatting
    ax1.set_title("Edge-AI Pareto Frontier: Computational Workload vs. Clinical Fidelity", fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel(r"Total Computational Workload (GFLOPs) $\downarrow$", fontsize=14, fontweight='bold')
    ax1.set_ylabel(r"PSNR (dB) $\uparrow$ [Solid Markers]", fontsize=14, fontweight='bold')
    ax2.set_ylabel(r"SSIM $\uparrow$ [Hollow Markers]", fontsize=14, fontweight='bold')
    
    ax1.set_xlim(0.5, 6.0)
    ax1.set_ylim(24.0, 30.0)
    ax2.set_ylim(0.88, 0.96)
    
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Custom Legend Box
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='PSNR (Primary Axis)'),
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='gray', markerfacecolor='none', markersize=10, linestyle='--', label='SSIM (Secondary Axis)'),
        plt.Line2D([0], [0], marker='none', color='w', label='Bubble Area $\propto$ Parameter Count')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True, facecolor='white', edgecolor='black')

    plt.tight_layout()
    plt.savefig("fig_pareto_advanced.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'fig_pareto_advanced.png'")

if __name__ == "__main__":
    plot_advanced_pareto()