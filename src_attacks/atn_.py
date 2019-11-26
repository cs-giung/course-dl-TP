import torch
import torch.nn as nn
import torch.optim as optim
from .atn_nets import _conv_fc


class P_ATN():

    def __init__(self, model, epsilon=8*4/255, weight=None, device='cuda'):

        self.model = model
        self.epsilon = epsilon
        self.device = device

        self.net = _conv_fc().to(device)
        if weight is not None:
            state_dict = torch.load(weight, map_location=device)
            self.net.load_state_dict(state_dict)


    def train(self, images, labels, learning_rate=1e-3):

        criterion_x = nn.MSELoss()
        criterion_y = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        images = images.to(self.device)
        labels = labels.to(self.device)

        perturbation = self.net(images)
        perturbation = perturbation.mul(self.epsilon)
        images_adv = images + perturbation
        images_adv = torch.clamp(images_adv, min=-2, max=2)
        lossX = criterion_x(images_adv, images)

        # outputs = self.model(images)
        outputs_adv = self.model(images_adv)
        self.model.zero_grad()
        lossY = criterion_y(outputs_adv, labels)

        loss = -lossY

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l2s = []
        for i in range(images.size(0)):
            l2s.append(torch.norm(images[i] - images_adv[i], p=2).item())

        return loss.item(), lossX.item(), lossY.item(), sum(l2s) / len(l2s)


    def perturb(self, images):
        images = images.to(self.device)
        perturbation = self.net(images)
        perturbation = perturbation.mul(self.epsilon)
        images_adv = images + perturbation
        images_adv = torch.clamp(images_adv, min=-2, max=2)
        return images_adv.detach()
