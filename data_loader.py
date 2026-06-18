import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import os

class LoDoPaBStreamer(Dataset):
    """
    Streams LoDoPaB-CT data directly from HDF5 files to avoid RAM exhaustion.
    Provides the sinogram (y) and the ground truth clean image (x).
    """
    def __init__(self, h5_file_path):
        super().__init__()
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"Dataset not found at {h5_file_path}. Are you running this in Kaggle?")
            
        self.h5_file_path = h5_file_path
        # We don't open the H5 file in __init__ to allow PyTorch DataLoader multi-processing
        self.file = None 

    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_file_path, 'r')
            self.observations = self.file['observation']
            self.ground_truths = self.file['ground_truth']

    def __len__(self):
        self._open_file()
        return self.observations.shape[0]

    def __getitem__(self, idx):
        self._open_file()
        
        # Extract and add the missing Channel dimension (C=1)
        y = torch.tensor(self.observations[idx], dtype=torch.float32).unsqueeze(0)
        x = torch.tensor(self.ground_truths[idx], dtype=torch.float32).unsqueeze(0)
        
        return y, x

def get_dataloaders(kaggle_input_dir, batch_size=4):
    """Initializes the training and validation streaming loaders."""
    train_path = os.path.join(kaggle_input_dir, 'ground_truth_train_000.hdf5') # Adjust based on actual Kaggle folder structure
    
    train_dataset = LoDoPaBStreamer(train_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    return train_loader