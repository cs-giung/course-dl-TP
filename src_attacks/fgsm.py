import torch
import torch.nn as nn


class FGSM():

    def __init__(self, model, step_size=4/255, num_steps=8):

        self.model = model
        self.step_size = step_size
        self.num_steps = num_steps
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, images, labels):

        x = images.clone()

        x.requires_grad_()
        outputs = self.model(x)
        self.model.zero_grad()
        loss = self.criterion(outputs, labels)
        loss.backward()
        x = x + self.num_steps * self.step_size * x.grad.sign()
        x = torch.clamp(x, min=-2, max=2)
        
        return x.detach()
