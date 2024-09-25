import torch
from torch.utils.data import Dataset

class SpatiotemporalTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
