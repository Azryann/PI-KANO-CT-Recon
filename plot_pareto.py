import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_frontier():
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
    
    # Data: [Theoretical GFLOPs, PSNR, Parameters]
    models = {
        "PAUM (CNN)": [112.4, 25.59, 30436],
        "JotlasNet (ViT)": [185.2, 26.99, 53122],
        "FDA-Net (Ours)": [1.09, 25.17, 8476]
    }
    
    colors = {"PAUM (CNN)": "#1f77b4", "JotlasNet (ViT)": "#2ca02c", "FDA-Net (Ours)": "#d62728"}
    markers = {"PAUM (CNN)": "o", "JotlasNet (ViT)": "s", "FDA-Net (Ours)": "*"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Edge-AI Viability Zone
    ax.axvspan(0.1, 5.0, color='#c6efce', alpha=0.4, label='Edge-AI Viability Zone (< 5.0 GFLOPs)')
    
    for name, (gflops, psnr, params) in models.items():
        bubble_size = params / 40  
        ax.scatter(gflops, psnr, s=bubble_size, c=colors[name], marker=markers[name], 
                   edgecolors='black', linewidth=1.5, alpha=0.8, label=f"{name} ({params//1000}K params)")
        
        offset = (15, -15) if "PAUM" in name else (15, 10)
        ax.annotate(name, (gflops, psnr), xytext=offset, textcoords='offset points', 
                    fontsize=12, fontweight='bold', color=colors[name])

    # True Pareto Frontier Line (Connects FDA-Net and JotlasNet)
    pareto_x = [1.09, 185.2]
    pareto_y = [25.17, 26.99]
    ax.plot(pareto_x, pareto_y, 'k--', alpha=0.6, linewidth=2, label='Pareto Frontier')

    ax.set_xscale('log')
    ax.set_title("Edge-AI Pareto Frontier: Reconstruction Quality vs. Computational Workload", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Total Computational Workload (Log10 GFLOPs) ↓", fontsize=14)
    ax.set_ylabel("Clinical Fidelity (HU-Windowed PSNR) ↑", fontsize=14)
    
    ax.set_xlim(0.5, 300)
    ax.set_ylim(24.5, 27.5)
    ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    
    ax.text(0.6, 27.3, "Bubble Area $\propto$ Parameter Count", fontsize=12, style='italic', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig("fig_pareto_frontier.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'fig_pareto_frontier.png'")

if __name__ == "__main__":
    plot_pareto_frontier()