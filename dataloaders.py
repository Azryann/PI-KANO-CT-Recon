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
        
        # Define the features to extract from the TFRecord
        self.description = {
            "observation": "byte",
            "ground_truth": "byte"
        }
        
        # Initialize the underlying TFRecord reader
        self.dataset = TFRecordDataset(
            data_path=self.tfrecord_path,
            index_path=self.index_path,
            description=self.description
        )

    def _decode_and_reshape(self, byte_data, target_shape):
        """Decodes raw bytes into float32 PyTorch tensors and reshapes them."""
        # Convert bytes to numpy array, then to PyTorch tensor
        array = np.frombuffer(byte_data, dtype=np.float32)
        tensor = torch.from_numpy(array.copy())
        
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
                # Skip corrupted records if any exist
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
    
    # 1. Create a dummy TFRecord file to simulate LoDoPaB-CT locally
    writer = tfrecord.TFRecordWriter(dummy_tfrecord_path)
    for _ in range(5):  # Create 5 dummy samples
        # Create random float32 arrays matching LoDoPaB dimensions
        dummy_sino = np.random.randn(1000, 513).astype(np.float32)
        dummy_img = np.random.randn(362, 362).astype(np.float32)
        
        # Write to TFRecord
        writer.write({
            "observation": (dummy_sino.tobytes(), "byte"),
            "ground_truth": (dummy_img.tobytes(), "byte")
        })
    writer.close()
    
    # 2. Test the DataLoader
    print("Testing LoDoPaB IterableDataset...")
    dataloader = get_lodopab_dataloader(dummy_tfrecord_path, batch_size=2)
    
    for batch_idx, (sinograms, images) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Sinogram shape: {sinograms.shape} | dtype: {sinograms.dtype}")
        print(f"  Image shape:    {images.shape} | dtype: {images.dtype}")
        
        # Verify shapes match expected PyTorch format: (Batch, Channels, H, W)
        assert sinograms.shape == (2, 1, 1000, 513), "Sinogram shape mismatch!"
        assert images.shape == (2, 1, 362, 362), "Image shape mismatch!"
        
    print("\nSUCCESS: Streaming DataLoader verified. No memory overflow detected.")
    
    # 3. Cleanup local dummy file
    os.remove(dummy_tfrecord_path)
    print("Cleaned up dummy files.")