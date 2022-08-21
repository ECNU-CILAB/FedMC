import torch
from torch import nn


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.shared_clf = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.shared_clf(x)
        return x