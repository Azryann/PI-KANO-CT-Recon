import os
import torch
from torch.utils.data import IterableDataset, DataLoader

# Safely import TensorFlow for Kaggle's native TFRecord parsing
try:
    import tensorflow as tf
    # CRITICAL: Hide GPUs from TensorFlow so it doesn't steal PyTorch's VRAM!
    tf.config.set_visible_devices([], 'GPU')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class KaggleLoDoPaBDataset(IterableDataset):
    """
    Native TensorFlow parser for LoDoPaB-CT wrapped in a PyTorch IterableDataset.
    Perfectly decodes `tf.io.serialize_tensor` headers.
    """
    def __init__(self, tfrecord_path):
        super().__init__()
        self.tfrecord_path = tfrecord_path

    def __iter__(self):
        # Native TF Dataset
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        
        # The exact schema used by the dataset authors
        feature_description = {
            'observation': tf.io.FixedLenFeature([], tf.string),
            'ground_truth': tf.io.FixedLenFeature([], tf.string)
        }

        def _parse_function(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            
            # Native decoding of the proprietary byte header
            obs = tf.io.parse_tensor(parsed['observation'], out_type=tf.float32)
            gt = tf.io.parse_tensor(parsed['ground_truth'], out_type=tf.float32)
            
            # Reshape to standard PyTorch format: (Channels, H, W)
            obs = tf.reshape(obs, [1, 1000, 513])
            gt = tf.reshape(gt, [1, 362, 362])
            return obs, gt

        parsed_dataset = raw_dataset.map(_parse_function)

        # Stream directly into PyTorch
        for obs, gt in parsed_dataset:
            # Convert TF Tensor -> NumPy Array -> PyTorch Tensor
            yield torch.from_numpy(obs.numpy()), torch.from_numpy(gt.numpy())

class LocalDummyDataset(IterableDataset):
    """Fallback for local testing without TensorFlow or the 85GB file."""
    def __init__(self, batch_size):
        self.batch_size = batch_size
    def __iter__(self):
        for _ in range(5): # Yield 5 dummy samples
            yield torch.randn(1, 1000, 513), torch.randn(1, 362, 362)

def get_lodopab_dataloader(tfrecord_path, batch_size=4, num_workers=0):
    """Factory function to create the correct DataLoader."""
    if TF_AVAILABLE and os.path.exists(tfrecord_path):
        dataset = KaggleLoDoPaBDataset(tfrecord_path)
    else:
        print("[INFO] Using Local Dummy Dataset (TF or file missing).")
        dataset = LocalDummyDataset(batch_size)
        
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    # Test initialization locally
    print("Testing Dataloader Initialization...")
    dl = get_lodopab_dataloader("dummy.tfrecord", batch_size=2)
    for batch_idx, (sino, img) in enumerate(dl):
        print(f"Batch {batch_idx+1} | Sino: {sino.shape} | Img: {img.shape}")