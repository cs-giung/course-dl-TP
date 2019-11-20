"""
python demo_pgd.py --pgd_type linf
python demo_pgd.py --pgd_type l2

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

from src import get_test_loader, IND2CLASS
from src import VGG, PGD_Linf, PGD_L2


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
    plt.yticks(np.arange(len(soft1)), IND2CLASS)
    fig.add_subplot(223)
    plt.imshow(img2)
    fig.add_subplot(224)
    plt.barh(np.arange(len(soft2)), soft2)
    plt.xlim(0, 1)
    plt.yticks(np.arange(len(soft2)), IND2CLASS)
    plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--pgd_type', default=None, type=str)
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
    test_dataloader = get_test_loader()

    # PGD instance
    if args.pgd_type == 'linf':
        PGD = PGD_Linf(model=net)
    elif args.pgd_type == 'l2':
        PGD = PGD_L2(model=net)

    # PGD examples
    for images, labels in test_dataloader:

        images = images.to(device)
        images_adv = PGD.perturb(images, labels)

        outputs = net(images)
        outputs_adv = net(images_adv)

        for image, image_adv, output, output_adv in zip(images, images_adv, outputs, outputs_adv):

            img = recover_image(image)
            soft_label = F.softmax(output, dim=0).cpu().detach().numpy()

            img_adv = recover_image(image_adv)
            soft_label_adv = F.softmax(output_adv, dim=0).cpu().detach().numpy()

            l2_dist = torch.norm(image - image_adv, 2).item()
            linf_dist = torch.norm(image - image_adv, float('inf')).item()
            print('%s -> %s' % (IND2CLASS[np.argmax(soft_label)], IND2CLASS[np.argmax(soft_label_adv)]))
            print('l2   dist = %.4f' % l2_dist)
            print('linf dist = %.4f' % linf_dist)
            print()

            plot_comparison(img, img_adv, soft_label, soft_label_adv)
            plt.show()

        break


if __name__ == '__main__':
    main()
