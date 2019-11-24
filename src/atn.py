import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class _atn_net(nn.Module):

    def __init__(self):
        super(_atn_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, 1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.layers(x)
        out = 2 * x
        return out


class AAE_ATN():


    def __init__(self, device, weight=None, target_classifier=None):

        self.device = device
        self.target_classifier = target_classifier

        self.net = _atn_conv().to(device)
        if weight is not None:
            state_dict = torch.load(weight, map_location=device)
            self.net.load_state_dict(state_dict)


    def _reranking(self, soft_labels, alpha):

        for idx in range(soft_labels.size(0)):
            _, ind_max = soft_labels[idx].max(0)
            _, ind_min = soft_labels[idx].min(0)
            soft_labels[idx][ind_max.item()] = alpha * soft_labels[idx][ind_min.item()]
        soft_labels_norm = torch.norm(soft_labels, dim=1, keepdim=True)
        soft_labels = soft_labels.div(soft_labels_norm)
        return soft_labels


    def train(self, images, alpha=0.1, beta=0.7, learning_rate=0.001):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        images = images.to(self.device)
        images_adv = self.net(images)
        lossX = criterion(images_adv, images)

        outputs = self.target_classifier(images)
        outputs_adv = self.target_classifier(images_adv)

        soft_labels = F.softmax(outputs, dim=1)
        soft_labels_adv = F.softmax(outputs_adv, dim=1)

        soft_labels_reranked = self._reranking(soft_labels, alpha).detach()
        lossY = criterion(soft_labels_adv, soft_labels_reranked)

        loss = beta * lossX + (1 - beta) * lossY

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l2s = []
        for i in range(images.size(0)):
            l2s.append(torch.norm(images[i] - images_adv[i], p=2).item())

        return loss.item(), lossX.item(), lossY.item(), sum(l2s) / len(l2s)


    def perturb(self, images, threshold=None):

        if threshold is None:
            return self.net(images).detach()

        images = images.to(self.device)
        images_adv = self.net(images).detach()

        l2s = []
        for i in range(images.size(0)):
            l2s.append(torch.norm(images[i] - images_adv[i], p=2).item())

        if sum(l2s) / len(l2s) > threshold:
            return images, False
        else:
            return images_adv, True
