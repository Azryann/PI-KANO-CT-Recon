import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_frontier():
    # Data: [GFLOPs, PSNR, Parameters]
    models = {
        "PAUM (CNN)": [5.13, 25.59, 30436],
        "JotlasNet (ViT)": [1.58, 26.99, 53122],
        "FDA-Net (Ours)": [1.09, 25.17, 8476]
    }
    
    colors = {"PAUM (CNN)": "#1f77b4", "JotlasNet (ViT)": "#2ca02c", "FDA-Net (Ours)": "#d62728"}
    markers = {"PAUM (CNN)": "o", "JotlasNet (ViT)": "s", "FDA-Net (Ours)": "*"}
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Draw the "Edge-AI Viability Zone" (e.g., < 2.0 GFLOPs)
    plt.axvspan(0, 2.0, color='#c6efce', alpha=0.4, label='Edge-AI Viability Zone (< 2.0 GFLOPs)')
    
    for name, (gflops, psnr, params) in models.items():
        # Bubble size proportional to parameter count
        bubble_size = params / 50  
        
        plt.scatter(gflops, psnr, s=bubble_size, c=colors[name], marker=markers[name], 
                    edgecolors='black', linewidth=1.5, alpha=0.8, label=f"{name} ({params//1000}K params)")
        
        # Annotate the bubbles
        plt.annotate(name, (gflops, psnr), xytext=(10, 10), textcoords='offset points', 
                     fontsize=12, fontweight='bold', color=colors[name])

    # Draw Pareto Frontier Line
    pareto_x = [1.09, 1.58]
    pareto_y = [25.17, 26.99]
    plt.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')

    plt.title("Edge-AI Pareto Frontier: Reconstruction Quality vs. Computational Workload", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Total Computational Workload (GFLOPs) ↓", fontsize=14)
    plt.ylabel("Clinical Fidelity (HU-Windowed PSNR) ↑", fontsize=14)
    
    plt.xlim(0.5, 6.0)
    plt.ylim(24.5, 27.5)
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    
    # Add a note about bubble size
    plt.text(0.6, 27.3, "Bubble Area $\propto$ Parameter Count", fontsize=12, style='italic', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig("fig_pareto_frontier.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'fig_pareto_frontier.png'")

if __name__ == "__main__":
    plot_pareto_frontier()