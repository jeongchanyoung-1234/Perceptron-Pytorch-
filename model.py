import torch
import torch.nn as nn

class IrisClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size):
        super(IrisClassifier, self).__init__()
        self.W_in = nn.Linear(input_size, hidden_size, dtype=torch.float)
        self.W_out = nn.Linear(hidden_size, 3, dtype=torch.float)

    def forward(self, x):
        z = self.W_in(x)
        y = self.W_out(z)
        return y

