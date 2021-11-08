import argparse

import torch.nn as nn
import torch.optim as optim

from data import get_dataloader
from model import IrisClassifier
from trainer import IrisTrainer

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--lr', type=float, default=2e-2)
    p.add_argument('--verbose', type=int, default=10)

    config = p.parse_args()

    return config

def train(config):
    model = IrisClassifier(input_size=4,
                           hidden_size=config.hidden_size,
                           n_layers=config.n_layers)

    optimizer_map = {'adam': optim.Adam,
                     'sgd': optim.SGD,
                     'rmsprop': optim.RMSprop}

    optimizer = optimizer_map[config.optimizer.lower()](model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    train_dataloader, valid_dataloader = get_dataloader(config)

    trainer = IrisTrainer(
        config,
        model,
        optimizer,
        criterion,
        train_dataloader,
        valid_dataloader)

    trainer.train()

if __name__ == '__main__':
    config = define_argparser()
    train(config)