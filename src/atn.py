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

    def __init__(self, device, beta=0.99, target_classifier=None):

        self.device = device
        self.beta = beta
        self.target_classifier = target_classifier

        self.net = _atn_conv().to(self.device)
        self.loss_fn1 = nn.MSELoss()
        self.loss_fn2 = nn.CrossEntropyLoss()

    def train(self, train_loader, learning_rate):

        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        for idx, (images, labels) in enumerate(train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            images_adv = self.net(images)
            loss1 = self.loss_fn1(images_adv, images)

            outputs_adv = self.target_classifier(images_adv)
            loss2 = -self.loss_fn2(outputs_adv, labels)

            loss = self.beta * loss1 + (1 - self.beta) * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def perturb(self, images):
        images = images.to(self.device)
        return self.net(images).detach()
