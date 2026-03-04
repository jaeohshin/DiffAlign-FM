import pickle
import torch
from torch.utils.data import Dataset

class ConformationDataset(Dataset):
    def __init__(self, data_path, transform=None):
        print(f"Loading data from {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} conformations")
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
