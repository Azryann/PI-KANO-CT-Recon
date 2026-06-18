import torch
from torch.utils.data import Dataset, DataLoader
from tfrecord.torch.dataset import TFRecordDataset

class LoDoPaBStreamer(Dataset):
    """
    Streams LoDoPaB-CT data from TFRecord files.
    """
    def __init__(self, tfrecord_path, index_path):
        # TFRecordDataset requires an index file to allow for shuffling/indexing
        self.dataset = TFRecordDataset(tfrecord_path, index_path, description={
            "observation": "float32",
            "ground_truth": "float32"
        })

    def __len__(self):
        # We define length based on the dataset size; 
        # for LoDoPaB full training, this is typically ~35,000+
        return 35820 

    def __getitem__(self, idx):
        # Retrieve record
        data = next(iter(self.dataset))
        y = torch.tensor(data['observation']).unsqueeze(0)
        x = torch.tensor(data['ground_truth']).unsqueeze(0)
        return y, x

def get_dataloaders(tfrecord_path, index_path, batch_size=4):
    dataset = LoDoPaBStreamer(tfrecord_path, index_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)