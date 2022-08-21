import torch.nn as nn
import torch.nn.functional as F


class HAR(nn.Module):
    def __init__(self):
        super(HAR, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Flatten()
        )

        self.shared_clf = nn.Sequential(
            nn.Linear(1184, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.shared_clf(x)
        return x
