import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim

# Import our custom PI-KANO modules
from dataloaders import get_lodopab_dataloader
from pi_kano import PI_KANO

def compute_metrics(pred, gt):
    """
    Computes PSNR, NRMSE, and SSIM between the reconstructed image and the ground truth.
    Supports Batch dimensions.
    """
    pred_np = pred.detach().cpu().squeeze().numpy()
    gt_np = gt.detach().cpu().squeeze().numpy()
    
    # Handle single sample vs batch arrays
    if len(pred_np.shape) == 2:
        pred_np = [pred_np]
        gt_np = [gt_np]
        
    psnrs, nrmses, ssims = [], [], []
    
    for p, g in zip(pred_np, gt_np):
        # Calculate NRMSE
        rmse = np.sqrt(np.mean((p - g) ** 2))
        nrmse = rmse / (np.mean(g) + 1e-8)
        nrmses.append(nrmse)
        
        # Calculate PSNR
        data_range = g.max() - g.min() + 1e-8
        psnr = 20 * np.log10(data_range / (rmse + 1e-8))
        psnrs.append(psnr)
        
        # Calculate SSIM
        s_val = ssim(p, g, data_range=data_range)
        ssims.append(s_val)
        
    return np.mean(psnrs), np.mean(nrmses), np.mean(ssims)


def train_pi_kano(tfrecord_path, epochs=1, batch_size=2, lr=1e-3, device='cuda'):
    print(f"Setting up PI-KANO Training on {device}...")
    
    # 1. Initialize Dataset & Dataloader
    dataloader = get_lodopab_dataloader(tfrecord_path, batch_size=batch_size)
    
    # 2. Instantiate Model
    # Dimensions match LoDoPaB-CT dataset
    model = PI_KANO(img_size=362, num_angles=1000, num_detectors=513, device=device).to(device)
    
    # 3. Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss scaling factor for physical consistency
    lambda_physics = 0.1
    
    model.train()
    print("Beginning Training Loop...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_psnr, epoch_nrmse, epoch_ssim = 0.0, 0.0, 0.0
        steps = 0
        
        for batch_idx, (sinograms, ground_truths) in enumerate(dataloader):
            # Move to device
            sinograms = sinograms.to(device)
            ground_truths = ground_truths.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Reconstruct CT image
            reconstructions = model(sinograms)
            
            # Loss Component 1: Image Alignment Loss (supervised ground-truth matching)
            alignment_loss = F.mse_loss(reconstructions, ground_truths)
            
            # Loss Component 2: Data Fidelity Loss (Enforces Physics Consistency)
            # Re-project the reconstruction back to measurement space
            pred_sinograms = model.physics(reconstructions)
            fidelity_loss = F.mse_loss(pred_sinograms, sinograms)
            
            # Fused Physics-Informed Loss
            total_loss = alignment_loss + lambda_physics * fidelity_loss
            
            # Backward pass & update weights
            total_loss.backward()
            optimizer.step()
            
            # Compute evaluation metrics for monitoring
            psnr_val, nrmse_val, ssim_val = compute_metrics(reconstructions, ground_truths)
            
            epoch_loss += total_loss.item()
            epoch_psnr += psnr_val
            epoch_nrmse += nrmse_val
            epoch_ssim += ssim_val
            steps += 1
            
            # Print status every step (locally) or every 50 steps (on Kaggle)
            if batch_idx % 1 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}] "
                      f"| Loss: {total_loss.item():.6f} | PSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f}")
                
            # Local prototyping protection (break early after 3 batches)
            if device == 'cuda' and torch.cuda.get_device_properties(device).total_memory / (1024**3) < 5.0:
                if steps >= 3:
                    print("\n[INFO] Local Prototyping Complete (Stopped early to prevent RTX 3050 OOM).")
                    break
        
        # Calculate Epoch Averages
        avg_loss = epoch_loss / steps
        avg_psnr = epoch_psnr / steps
        avg_nrmse = epoch_nrmse / steps
        avg_ssim = epoch_ssim / steps
        
        print(f"\n=== Epoch {epoch+1} Summary ===")
        print(f"Average Loss:  {avg_loss:.6f}")
        print(f"Average PSNR:  {avg_psnr:.2f} dB")
        print(f"Average NRMSE: {avg_nrmse:.4f}")
        print(f"Average SSIM:  {avg_ssim:.4f}\n")


if __name__ == "__main__":
    import numpy as np
    import tfrecord
    
    # Check if we are running in Kaggle Cloud or Local
    KAGGLE_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(KAGGLE_PATH):
        print("Kaggle environment verified. Commencing full-scale training on 85GB LoDoPaB...")
        train_pi_kano(KAGGLE_PATH, epochs=5, batch_size=4, lr=1e-3, device=device)
    else:
        print("Kaggle dataset not found. Generating a temporary dummy dataset for local prototyping...")
        dummy_path = "local_temp_lodopab.tfrecord"
        
        # Write dummy samples
        writer = tfrecord.TFRecordWriter(dummy_path)
        for _ in range(4):  # Generate 4 dummy samples
            dummy_sino = np.random.randn(1000, 513).astype(np.float32)
            dummy_img = np.random.randn(362, 362).astype(np.float32)
            writer.write({
                "observation": (dummy_sino.tobytes(), "byte"),
                "ground_truth": (dummy_img.tobytes(), "byte")
            })
        writer.close()
        
        try:
            # Run local prototyping loop
            train_pi_kano(dummy_path, epochs=1, batch_size=2, lr=1e-3, device=device)
        finally:
            # Cleanup
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
                print("Cleaned up temporary local files.")