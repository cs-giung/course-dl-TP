import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(2*3*32*32, 2*3*32*32)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(1, -1)
        x = self.fc(x)
        x = x.view(2, 3, 32, 32)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    net = AutoEncoder()
    summary(net, (3, 32, 32))
