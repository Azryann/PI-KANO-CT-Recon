import os
import torch
import numpy as np
import h5py
from torch.utils.data import IterableDataset, Dataset, DataLoader

# Safely import TensorFlow for Kaggle's native TFRecord parsing (LoDoPaB)
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ==========================================
# 1. LoDoPaB-CT Dataset (TFRecord Iterable)
# ==========================================
class KaggleLoDoPaBDataset(IterableDataset):
    def __init__(self, tfrecord_path):
        super().__init__()
        self.tfrecord_path = tfrecord_path

    def __iter__(self):
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        feature_description = {
            'observation': tf.io.FixedLenFeature([], tf.string),
            'ground_truth': tf.io.FixedLenFeature([], tf.string)
        }

        def _parse_function(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            obs = tf.io.parse_tensor(parsed['observation'], out_type=tf.float32)
            gt = tf.io.parse_tensor(parsed['ground_truth'], out_type=tf.float32)
            obs = tf.reshape(obs, [1, 1000, 513])
            gt = tf.reshape(gt, [1, 362, 362])
            return obs, gt

        parsed_dataset = raw_dataset.map(_parse_function)
        for obs, gt in parsed_dataset:
            yield torch.from_numpy(obs.numpy()), torch.from_numpy(gt.numpy())

# ==========================================
# 2. Mayo Clinic LDCT Dataset (HDF5 / Numpy)
# ==========================================
class MayoClinicDataset(Dataset):
    """
    Standard PyTorch Dataset for Mayo Clinic LDCT Grand Challenge.
    Assumes data has been pre-processed into a single .h5 file containing 
    'sinogram' and 'ground_truth' datasets.
    """
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        
        # Open file to get length
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['sinogram'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open file dynamically for multi-processing safety
        with h5py.File(self.h5_path, 'r') as f:
            # Mayo Clinic data shapes depend on the specific scanner geometry used.
            # We dynamically extract the shape and add the channel dimension (C=1).
            sino = f['sinogram'][idx].astype(np.float32)
            img = f['ground_truth'][idx].astype(np.float32)
            
        return torch.from_numpy(sino).unsqueeze(0), torch.from_numpy(img).unsqueeze(0)

# ==========================================
# 3. Local Dummy Dataset (For 4GB VRAM Testing)
# ==========================================
class LocalDummyDataset(IterableDataset):
    def __init__(self, dataset_type='lodopab'):
        self.dataset_type = dataset_type
        
    def __iter__(self):
        for _ in range(5):
            if self.dataset_type == 'lodopab':
                yield torch.randn(1, 1000, 513), torch.randn(1, 362, 362)
            elif self.dataset_type == 'mayo':
                # Example Mayo Clinic Dimensions (e.g., Siemens SOMATOM)
                yield torch.randn(1, 736, 736), torch.randn(1, 512, 512)

# ==========================================
# Factory Function
# ==========================================
def get_ct_dataloader(dataset_name, data_path, batch_size=2, num_workers=0):
    """
    Modular factory function to load the correct dataset.
    dataset_name: 'lodopab' or 'mayo'
    """
    print(f"Initializing DataLoader for: {dataset_name.upper()}")
    
    if not os.path.exists(data_path):
        print(f"[WARNING] Data path '{data_path}' not found. Using local dummy streaming.")
        dataset = LocalDummyDataset(dataset_type=dataset_name)
    else:
        if dataset_name == 'lodopab':
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is required to parse LoDoPaB .tfrecord files.")
            dataset = KaggleLoDoPaBDataset(data_path)
        elif dataset_name == 'mayo':
            dataset = MayoClinicDataset(data_path)
        else:
            raise ValueError("dataset_name must be 'lodopab' or 'mayo'")
            
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    # Test modularity locally
    print("--- Testing Modularity ---")
    
    # Test LoDoPaB Pipeline
    dl_lodo = get_ct_dataloader('lodopab', 'dummy_path.tfrecord', batch_size=1)
    for sino, img in dl_lodo:
        print(f"LoDoPaB Shape -> Sino: {sino.shape} | Img: {img.shape}")
        break
        
    # Test Mayo Clinic Pipeline
    dl_mayo = get_ct_dataloader('mayo', 'dummy_path.h5', batch_size=1)
    for sino, img in dl_mayo:
        print(f"Mayo Clinic Shape -> Sino: {sino.shape} | Img: {img.shape}")
        break