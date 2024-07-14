from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sequence_data, targets):
        self.sequence_data = sequence_data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequence_data[idx], self.targets[idx]