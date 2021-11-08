import torch
import torch.nn as nn

class IrisClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers):
        super(IrisClassifier, self).__init__()
        in_layers = [nn.Linear(input_size, hidden_size), nn.ReLU()] +\
                    [nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_layers - 1)

        self.W_in = nn.Sequential(*in_layers)
        self.W_out = nn.Linear(hidden_size, 3)

    def forward(self, x):
        z = self.W_in(x)
        y = self.W_out(z)
        return y

