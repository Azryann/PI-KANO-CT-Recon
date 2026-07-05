import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloaders import get_ct_dataloader
from fda_net import FDA_Net

def plot_attention_weights(data_path, device='cuda'):
    print("Extracting Frequency-Domain Attention Weights...")
    img_size, angles, detectors, phys_scale = 362, 1000, 513, 0.1
    
    model = FDA_Net(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    ckpt_path = "FDA_Net_subset_BEST.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model_state'])
        model.eval()
    else:
        print("FDA-Net weights not found!")
        return

    dataloader = get_ct_dataloader('lodopab', data_path, batch_size=1)
    
    # Hook to extract attention weights
    attention_weights = []
    def hook_fn(module, input, output):
        # output shape is (B, C, 1, 1)
        attention_weights.append(output.detach().cpu().squeeze().numpy())
        
    # Register hook on the Sigmoid output of the first cascade's FDA module
    handle = model.blocks[0].freq_attn.mlp[-1].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for sino, _ in dataloader:
            _ = model(sino.to(device) / phys_scale)
            break # Just need one pass
            
    handle.remove()
    
    weights = attention_weights[0]
    channels = np.arange(len(weights))
    
    # Plotting
    plt.figure(figsize=(8, 4))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
    
    # Bar chart of attention weights per channel
    plt.bar(channels, weights, color='#d62728', alpha=0.8, edgecolor='black')
    
    # Add a trendline
    plt.plot(channels, weights, color='black', linestyle='--', alpha=0.5)
    
    plt.title("Learned Frequency-Domain Attention Weights (Cascade 1)", fontsize=14, fontweight='bold')
    plt.xlabel("Latent Frequency Channel Index", fontsize=12)
    plt.ylabel("Attention Weight $\\alpha \in [0, 1]$", fontsize=12)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig("fig_fda_attention.png", dpi=300)
    print("SUCCESS: Saved 'fig_fda_attention.png'")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        plot_attention_weights(LODOPAB_PATH, device=device)