import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Example dataset
data = torch.randn(100, 10)  # 100 samples, 10 features each
targets = torch.randint(0, 2, (100,))  # 100 target values (binary classification)

# Wrap data and target into a TensorDataset
dataset = TensorDataset(data, targets)

# DataLoader
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fetch a random batch
for batch in dataloader:
    inputs, labels = batch
    print(inputs, labels)
    break  # We break after fetching the first batch, which is randomly shuffled
