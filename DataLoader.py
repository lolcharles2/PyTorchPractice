import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        # Loading data
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])  # All columns apart from first one
        self.y = torch.from_numpy(xy[:, [0]])  # First column only
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# training loop
epochs = 2
tot_samples = len(dataset)
num_iterations = math.ceil(tot_samples / 4)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch + 1}/{epochs}, step: {i + 1}/{num_iterations}, inputs {inputs.shape}')
