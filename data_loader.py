import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tfrecord.torch.dataset import TFRecordDataset

class LoDoPaBStreamer(Dataset):
    def __init__(self, tfrecord_path, index_path):
        # Change type to 'byte' to match the binary format
        self.dataset = TFRecordDataset(tfrecord_path, index_path, description={
            "observation": "byte",
            "ground_truth": "byte"
        })

    def __len__(self):
        return 35820 

    def __getitem__(self, idx):
        # We need to access the dataset directly via the iterator
        data = next(iter(self.dataset))
        
        # Convert raw bytes back to numpy arrays
        # Assuming the data was saved as float32 in binary
        y_bytes = np.frombuffer(data['observation'], dtype=np.float32)
        x_bytes = np.frombuffer(data['ground_truth'], dtype=np.float32)
        
        # Reshape based on LoDoPaB standard dimensions
        # Sinograms: 1000 angles x 513 detectors
        # Images: 362 x 362
        y = torch.tensor(y_bytes.reshape(1, 1000, 513), dtype=torch.float32)
        x = torch.tensor(x_bytes.reshape(1, 362, 362), dtype=torch.float32)
        
        return y, x