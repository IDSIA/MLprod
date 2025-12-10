import torch.nn as nn
import torch


class Model(nn.Module):
    """This is the PyTorch definition of our model."""

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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
