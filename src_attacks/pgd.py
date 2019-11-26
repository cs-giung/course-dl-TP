import torch
import torch.nn as nn


class PGD_Linf():

    def __init__(self, model, epsilon=8*4/255, step_size=4/255, num_steps=10, random_start=True):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, images, labels):

        if self.random_start:
            x = images + (torch.rand_like(images) - 0.5) * 2 * self.epsilon
            x = torch.clamp(x, min=-2, max=2)
        else:
            x = images.clone()

        for _ in range(self.num_steps):

            x.requires_grad_()
            outputs = self.model(x)
            self.model.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()

            x = x + self.step_size * x.grad.sign()
            r = torch.clamp(x - images, min=-self.epsilon, max=self.epsilon).detach()

            x = torch.clamp(images + r, min=-2, max=2)
        
        return x.detach()


class PGD_L2():

    def __init__(self, model, epsilon=20*4/255, step_size=4/255, num_steps=10, random_start=True):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, images, labels):

        if self.random_start:
            x = images + (torch.rand_like(images) - 0.5) * 2 * self.epsilon
            x = torch.clamp(x, min=-2, max=2)
        else:
            x = images.clone()

        for _ in range(self.num_steps):

            x.requires_grad_()
            outputs = self.model(x)
            self.model.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()

            x = x + self.step_size * x.grad.sign()
            r = (x - images).view(x.size()[0], -1).detach()
            r_norm = torch.norm(r, dim=1, keepdim=True)
            # eps = torch.ones_like(r) * self.epsilon
            # eps_norm = torch.norm(eps, dim=1, keepdim=True)
            # mask = (r_norm < eps_norm).float()
            # r = torch.mul(r, mask) + torch.mul(eps_norm[0] * r / r_norm, 1 - mask)
            mask = (r_norm < self.epsilon).float()
            r = torch.mul(r, mask) + torch.mul(self.epsilon * r / r_norm, 1 - mask)
            r = r.view(x.shape)

            x = torch.clamp(images + r, min=-2, max=2)

        return x.detach()
