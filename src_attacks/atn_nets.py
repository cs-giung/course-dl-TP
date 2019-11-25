import torch.nn as nn


class _conv_fc(nn.Module):

    def __init__(self):
        super(_conv_fc, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 3072),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 3, 32, 32)
        return x
