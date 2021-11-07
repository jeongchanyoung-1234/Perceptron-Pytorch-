import torch
import torch.utils.data as data
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class IrisDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(config):
    X, y = load_iris(return_X_y=True)
    train_X, valid_X, train_y, valid_y = train_test_split(X, y,
                                                          train_size = .8,
                                                          shuffle=True)
    train_dataloader = data.DataLoader(
        IrisDataset(train_X, train_y),
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_dataloader = data.DataLoader(
        IrisDataset(valid_X, valid_y),
        batch_size=config.batch_size,
        shuffle=True
    )

    return train_dataloader, valid_dataloader