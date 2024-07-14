import torch
import numpy as np

from torch.utils.data import TensorDataset
class CombinedDataset(TensorDataset):
    def __init__(self, tabular_data, sequence_data, targets):
        self.tabular_data = tabular_data
        self.sequence_data = sequence_data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        tabular_row = self.tabular_data.iloc[idx].values.astype(np.float32)
        return torch.tensor(tabular_row), self.sequence_data[idx], self.targets[idx]