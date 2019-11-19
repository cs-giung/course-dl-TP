import torch
import torch.nn as nn
import torch.optim as optim


class _atn_conv(nn.Module):

    def __init__(self):
        super(_atn_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 3),
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
        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr=0.01, momentum=0.9,
                                   weight_decay=5e-4)

    def train(self, train_loader):

        for images, labels in train_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.net(images)

            loss1 = self.loss_fn1(outputs, images)
            loss2 = 1
            loss = 0.1 * loss1 + 0.9 + loss2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def perturb(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        return self.net(images).detach()
