import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define a simple custom dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
        return sample

# Generate some random data
data = np.random.randn(100, 3)  # 100 samples, each with 3 features
labels = np.random.randint(0, 2, 100)  # Binary labels for each sample

#breakpoint()

# Create an instance of MyDataset
dataset = MyDataset(data, labels)

# Create a DataLoader
batch_size = 10
#breakpoint()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    print("Data:", batch['data'])
    print("Labels:", batch['label'])
    print()  # Blank line for readability
