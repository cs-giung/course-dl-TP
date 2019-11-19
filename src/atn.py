import torch
import torch.nn as nn
import torch.optim as optim


class _atn_conv(nn.Module):

    def __init__(self):
        super(_atn_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, 3),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        out = 2 * x
        return out


class ATN():

    def __init__(self, device):

        self.device = device
        self.net = _atn_conv().to(self.device)

        self.loss_fn1 = nn.MSELoss()
        self.loss_fn2 = None

    def train(self, train_loader, learning_rate):

        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        for idx, (images, labels) in enumerate(train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.net(images)

            loss1 = self.loss_fn1(outputs, images)
            loss2 = 1
            loss = 0.1 * loss1 + 0.9 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def perturb(self, images):
        images = images.to(self.device)
        return self.net(images).detach()
