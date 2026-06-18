import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

class LoDoPaBStreamer(Dataset):
    """
    Streams LoDoPaB-CT data directly from separate observation and ground truth HDF5 files.
    """
    def __init__(self, observation_file_path, ground_truth_file_path):
        super().__init__()
        if not os.path.exists(observation_file_path) or not os.path.exists(ground_truth_file_path):
            raise FileNotFoundError(f"Files not found.\nObs: {observation_file_path}\nGT: {ground_truth_file_path}")
            
        self.obs_path = observation_file_path
        self.gt_path = ground_truth_file_path
        
        self.obs_file = None 
        self.gt_file = None

    def _open_files(self):
        # We open files lazily to prevent multiprocessing crashes in PyTorch DataLoaders
        if self.obs_file is None:
            self.obs_file = h5py.File(self.obs_path, 'r')
            self.gt_file = h5py.File(self.gt_path, 'r')
            
            # LoDoPaB stores the actual tensors under the key 'data'
            self.observations = self.obs_file['data']
            self.ground_truths = self.gt_file['data']

    def __len__(self):
        self._open_files()
        return self.observations.shape[0]

    def __getitem__(self, idx):
        self._open_files()
        
        # Extract and add the missing Channel dimension (C=1)
        # LoDoPaB images are 362x362, Sinograms are 1000 angles x 513 detectors
        y = torch.tensor(self.observations[idx], dtype=torch.float32).unsqueeze(0)
        x = torch.tensor(self.ground_truths[idx], dtype=torch.float32).unsqueeze(0)
        
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