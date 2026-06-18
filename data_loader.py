import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tfrecord.torch.dataset import TFRecordDataset

class LoDoPaBStreamer(Dataset):
    def __init__(self, tfrecord_path, index_path):
        # The key fix: change description to 'byte'
        self.dataset = TFRecordDataset(tfrecord_path, index_path, description={
            "observation": "byte",
            "ground_truth": "byte"
        })

    def __len__(self):
        return 35820 

    def __getitem__(self, idx):
        # Fetch the serialized example
        data = next(iter(self.dataset))
        
        # Convert raw bytes to numpy float32 arrays
        # np.frombuffer interprets the raw binary string as a float array
        y_np = np.frombuffer(data['observation'], dtype=np.float32)
        x_np = np.frombuffer(data['ground_truth'], dtype=np.float32)
        
        # Reshape to correct physics dimensions:
        # Observation (Sinogram): 1000 angles x 513 detectors
        # Ground Truth (Image): 362 x 362
        y = torch.tensor(y_np.reshape(1, 1000, 513), dtype=torch.float32)
        x = torch.tensor(x_np.reshape(1, 362, 362), dtype=torch.float32)
        
        return y, x

def get_dataloaders(kaggle_input_dir, batch_size=4):
    """Dynamically finds the first HDF5 chunks and initializes the loader."""
    
    # Use glob to recursively search for the first chunk of data
    obs_files = sorted(glob.glob(os.path.join(kaggle_input_dir, '**', 'observation_train_000.hdf5'), recursive=True))
    gt_files = sorted(glob.glob(os.path.join(kaggle_input_dir, '**', 'ground_truth_train_000.hdf5'), recursive=True))
    
    if not obs_files or not gt_files:
        raise FileNotFoundError(
            f"Could not find the LoDoPaB HDF5 files in {kaggle_input_dir}. "
            "Please verify the dataset is attached to the Kaggle notebook."
        )
        
    print(f"Found Observation file: {obs_files[0]}")
    print(f"Found Ground Truth file: {gt_files[0]}")
        
    train_dataset = LoDoPaBStreamer(obs_files[0], gt_files[0])
    # num_workers=2 and pin_memory=True accelerate GPU transfer times
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    return train_loader