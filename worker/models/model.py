import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_size=26):
        super(Model, self).__init__()
        
        self.layers = [
            nn.Linear(input_size, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        ]

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)
