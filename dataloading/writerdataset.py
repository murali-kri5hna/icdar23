import torch
from torch.utils.data import Dataset

class WriterDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        label = self.labels[idx]
        return torch.tensor(encoding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

