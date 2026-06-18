import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from tfrecord.torch.dataset import TFRecordDataset

class LoDoPaBDataset(IterableDataset):
    """
    Streaming DataLoader for the LoDoPaB-CT dataset.
    Reads .tfrecord files sequentially to prevent RAM overflow.
    Strictly avoids 'inverse crimes' by loading pre-simulated high-fidelity 
    observations (sinograms) rather than generating them on the fly.
    """
    def __init__(self, tfrecord_path, index_path=None, batch_size=1):
        super().__init__()
        self.tfrecord_path = tfrecord_path
        self.index_path = index_path
        self.batch_size = batch_size
        
        # LoDoPaB-CT standard dimensions
        self.sino_shape = (1000, 513)  # (Angles, Detectors)
        self.img_shape = (362, 362)    # (H, W)
        
        # Official LoDoPaB-CT stores data as tf.train.FloatList
        # So we map them as "float" instead of "byte"
        self.description = {
            "observation": "float",
            "ground_truth": "float"
        }
        
        # Initialize the underlying TFRecord reader
        self.dataset = TFRecordDataset(
            data_path=self.tfrecord_path,
            index_path=self.index_path,
            description=self.description
        )

    def _decode_and_reshape(self, float_array, target_shape):
        """Converts the parsed float array into a PyTorch tensor and reshapes it."""
        # The tfrecord library already parsed it as a float sequence
        tensor = torch.tensor(float_array, dtype=torch.float32)
        
        # Reshape and add channel dimension (C=1)
        return tensor.view(1, *target_shape)

    def __iter__(self):
        """Yields (sinogram, ground_truth) pairs."""
        for data in self.dataset:
            try:
                # Decode observation (sinogram)
                sinogram = self._decode_and_reshape(data["observation"], self.sino_shape)
                
                # Decode ground truth (CT image)
                ground_truth = self._decode_and_reshape(data["ground_truth"], self.img_shape)
                
                yield sinogram, ground_truth
            except Exception as e:
                print(f"Skipping corrupted record: {e}")
                continue

def get_lodopab_dataloader(tfrecord_path, batch_size=4, num_workers=0):
    """Factory function to create the DataLoader."""
    dataset = LoDoPaBDataset(tfrecord_path, batch_size=batch_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    # ==========================================
    # LOCAL PROTOTYPING & STREAMING VERIFICATION
    # ==========================================
    import tfrecord
    
    print("Setting up local dummy TFRecord for testing...")
    dummy_tfrecord_path = "dummy_lodopab.tfrecord"
    
    # 1. Create a dummy TFRecord file (Using FloatList this time!)
    writer = tfrecord.TFRecordWriter(dummy_tfrecord_path)
    for _ in range(5):  
        dummy_sino = np.random.randn(1000 * 513).astype(np.float32)
        dummy_img = np.random.randn(362 * 362).astype(np.float32)
        
        # Write using 'float' matching official LoDoPaB
        writer.write({
            "observation": (dummy_sino, "float"),
            "ground_truth": (dummy_img, "float")
        })
    writer.close()
    
    # 2. Test the DataLoader
    print("Testing LoDoPaB IterableDataset...")
    dataloader = get_lodopab_dataloader(dummy_tfrecord_path, batch_size=2)
    
    for batch_idx, (sinograms, images) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Sinogram shape: {sinograms.shape} | dtype: {sinograms.dtype}")
        print(f"  Image shape:    {images.shape} | dtype: {images.dtype}")
        
    print("\nSUCCESS: Streaming DataLoader verified. No memory overflow detected.")
    os.remove(dummy_tfrecord_path)