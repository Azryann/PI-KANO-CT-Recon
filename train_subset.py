import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN, KirchhoffPhysicsConstraint

def train_60_epochs_subset(data_path, device='cuda'):
    print(f"\n{'='*60}\nDAY 2 LAUNCH: 60-Epoch PI-KINN (20% Stratified Subset)\n{'='*60}")
    
    img_size, angles, detectors = 362, 1000, 513
    batch_size = 2
    
    # Load full dataset
    full_dataloader = get_ct_dataloader('lodopab', data_path, batch_size=batch_size)
    
    # 20% Subset: We manually break the epoch at 3,582 steps.
    steps_per_epoch = 3582 
    epochs = 60
    
    model = PI_KINN(img_size, angles, detectors, num_cascades=3, device=device).to(device)
    
    # FIX: Removed 'detectors' argument. Kirchhoff integral only needs img_size and angles.
    kirchhoff_op = KirchhoffPhysicsConstraint(img_size, angles, device=device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Hyperparameters from Project Plan Section 3
    lambda_data = 1.0
    lambda_physics = 0.1 
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, ep_l_data, ep_l_phys = 0.0, 0.0, 0.0
        start_time = time.time()
        
        data_iter = iter(full_dataloader)
        
        for step in range(steps_per_epoch):
            try:
                sinograms, ground_truths = next(data_iter)
            except StopIteration:
                break
                
            sinograms, ground_truths = sinograms.to(device), ground_truths.to(device)
            
            optimizer.zero_grad()
            reconstructions = model(sinograms)
            
            # L_reconstruction (Data Fidelity)
            l_data = F.mse_loss(reconstructions, ground_truths)
            
            # L_kirchhoff (Physics Constraint)
            k_out = kirchhoff_op(reconstructions)
            l_physics = F.mse_loss(k_out, sinograms.mean(dim=-1)) 
            
            loss = (lambda_data * l_data) + (lambda_physics * l_physics)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            ep_l_data += l_data.item()
            ep_l_phys += l_physics.item()
            
            if step % 500 == 0:
                print(f"Ep [{epoch+1}/{epochs}] Stp [{step}/{steps_per_epoch}] | Tot: {loss.item():.4f} | L_data: {l_data.item():.4f} | L_phys: {l_physics.item():.4f}")
                
        scheduler.step()
        
        # Save checkpoint
        torch.save({'epoch': epoch+1, 'model_state': model.state_dict()}, f"PI_KINN_subset_ep{epoch+1}.pth")
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Time: {time.time() - start_time:.1f}s | Avg L_data: {ep_l_data/steps_per_epoch:.4f} | Avg L_phys: {ep_l_phys/steps_per_epoch:.4f}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LODOPAB_PATH = "/kaggle/input/datasets/peeeeeg/lodopab/lodopab_full_dose_train.tfrecord"
    if os.path.exists(LODOPAB_PATH):
        train_60_epochs_subset(LODOPAB_PATH, device=device)