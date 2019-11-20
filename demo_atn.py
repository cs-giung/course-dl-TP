import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src import VGG
from src import ATN


ind2class = ('plane', 'car', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck')

def get_test_dataloader():

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.500, 0.500, 0.500),
            (0.250, 0.250, 0.250)
        )
    ])
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=test_transform
    )
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8)
    return test_dataloader


def recover_image(image):
    img = image.cpu().numpy()
    img = 0.250 * img + 0.500
    img = (img * 255).astype('uint8')
    img = np.clip(img, 0, 255)
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(img, 'RGB')
    return img


def plot_comparison(img1, img2, soft1, soft2):
    fig = plt.figure()
    fig.add_subplot(221)
    plt.imshow(img1)
    fig.add_subplot(222)
    plt.barh(np.arange(len(soft1)), soft1)
    plt.xlim(0, 1)
    plt.yticks(np.arange(len(soft1)), ind2class)
    fig.add_subplot(223)
    plt.imshow(img2)
    fig.add_subplot(224)
    plt.barh(np.arange(len(soft2)), soft2)
    plt.xlim(0, 1)
    plt.yticks(np.arange(len(soft2)), ind2class)
    plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    # settings
    device = args.device
    weight_path = './weights/vgg16_e086_90.62.pth'

    # classification model
    net = VGG('VGG16').to(device)
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # test dataset
    test_dataloader = get_test_dataloader()

    # train ATN
    atn = ATN(device=device, weight='./weights/base_atn_conv.pth', target_classifier=net)
    for epoch_idx in range(5):
        print(epoch_idx)
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            if batch_idx == 0:
                continue
            if batch_idx == 9:
                break
            atn.train(images, labels)

    # ATN examples
    for images, labels in test_dataloader:

        images = images.to(device)
        images_adv = atn.perturb(images)

        outputs = net(images)
        outputs_adv = net(images_adv)

        for image, image_adv, output, output_adv in zip(images, images_adv, outputs, outputs_adv):
            img = recover_image(image)
            soft_label = F.softmax(output, dim=0).cpu().detach().numpy()

            img_adv = recover_image(image_adv)
            soft_label_adv = F.softmax(output_adv, dim=0).cpu().detach().numpy()

            plot_comparison(img, img_adv, soft_label, soft_label_adv)
            plt.show()

        break


if __name__ == '__main__':
    main()
