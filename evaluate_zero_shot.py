import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image

from physics import RadonPhysics
from fda_net import FDA_Net
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate

def load_any_ct_image(image_path, img_size=362, device='cuda'):
    """ Loads any standard image, converts to grayscale, and resizes to our network dimension. """
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    # Resize to match our network architecture
    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    
    # Convert to tensor and normalize to [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Convert [0, 1] back to approximate mu-space (0.0 to 0.04) for the physics engine
    mu_tensor = torch.tensor(img_np * 0.04, dtype=torch.float32, device=device)
    return mu_tensor.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, H, W]

def run_zero_shot(image_path, device='cuda'):
    print(f"\n{'='*75}\nRunning Zero-Shot Generalization on Out-of-Distribution Data\n{'='*75}")
    
    img_size, angles, detectors = 362, 1000, 513
    physics = RadonPhysics(img_size, angles, detectors, device=device)
    
    # 1. Load the OOD Image (Ground Truth)
    gt_mu = load_any_ct_image(image_path, img_size, device)
    
    # 2. Simulate the CT Scanner (Generate Sinogram on the fly)
    print("Simulating CT acquisition...")
    with torch.no_grad():
        sinogram = physics.forward(gt_mu)
        # Add slight Poisson noise to simulate Low-Dose conditions
        noise = torch.randn_like(sinogram) * 0.05 * sinogram.max()
        noisy_sinogram = sinogram + noise

    # 3. Load Models
    models = {
        "PAUM": PAUM_Surrogate(img_size, angles, detectors, num_cascades=3, device=device).to(device),
        "JotlasNet": JotlasNet_Surrogate(img_size, angles, detectors, num_cascades=2, device=device).to(device),
        "FDA-Net (Ours)": FDA_Net(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    }
    
    for name, model in models.items():
        ckpt_name = name.split(" ")[0]
        ckpt_path = f"{ckpt_name}_subset_BEST.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model_state'])
            model.eval()
        else:
            print(f"[WARNING] {ckpt_path} not found!")

    # 4. Reconstruct
    results = {}
    with torch.no_grad():
        fbp_mu = physics.adjoint(noisy_sinogram)
        results["FBP"] = fbp_mu.squeeze().cpu().numpy()
        
        for name, model in models.items():
            pred_mu = model(noisy_sinogram)
            results[name] = pred_mu.squeeze().cpu().numpy()
            
    gt_np = gt_mu.squeeze().cpu().numpy()

    # 5. Plotting the Zero-Shot Figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ["Ground Truth (OOD)", "FBP", "PAUM", "JotlasNet", "FDA-Net (Ours)"]
    images = [gt_np, results["FBP"], results["PAUM"], results["JotlasNet"], results["FDA-Net (Ours)"]]
    
    for col in range(5):
        ax = axes[col]
        ax.imshow(images[col], cmap='bone')
        ax.set_title(titles[col], fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if col > 0:
            # Calculate simple SSIM for the plot
            s_val = ssim(images[col], gt_np, data_range=gt_np.max() - gt_np.min())
            ax.text(10, 340, f"SSIM: {s_val:.4f}", color='yellow', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig("fig_zero_shot.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'fig_zero_shot.png'")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find ANY CT image on Kaggle (e.g., from an AAPM or COVID-19 CT dataset)
    # Replace this path with the path to a .png or .jpg CT slice
    TEST_IMAGE_PATH = "/kaggle/input/some-ct-dataset/sample_image.png" 
    
    if os.path.exists(TEST_IMAGE_PATH):
        run_zero_shot(TEST_IMAGE_PATH, device=device)
    else:
        print(f"Please update TEST_IMAGE_PATH to point to a valid CT image.")