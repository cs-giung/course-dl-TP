import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src import get_test_loader, get_train_valid_loader
from src import VGG, ATN


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--atn_epoch', default=10, type=int)
    parser.add_argument('--atn_sample', default=0.1, type=float)
    args = parser.parse_args()

    # settings
    config = dict()
    config['device'] = args.device
    config['num_epoch'] = args.epochs
    config['batch_size'] = args.batch_size
    config['atn_epoch'] = args.atn_epoch
    config['atn_sample'] = args.atn_sample
    weight_path = './weights/vgg16_e086_90.62.pth'

    # classification model
    net = VGG('VGG16').to(config['device'])
    state_dict = torch.load(weight_path, map_location=config['device'])
    net.load_state_dict(state_dict)
    net.eval()

    # test dataset
    train_loader, _ = get_train_valid_loader(batch_size=32)

    # train ATN
    atn = ATN(device=config['device'],
              weight='./weights/base_atn_conv.pth',
              beta=0.99, target_classifier=net)

    for epoch_idx in range(1, config['atn_epoch'] + 1):
        losses = []
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx == int(config['atn_sample'] * len(train_loader)):
                break
            loss = atn.train(images, labels)
            losses.append(loss)
        print('[%3d / %3d] Avg. Loss: %f' % (epoch_idx, config['atn_epoch'], sum(losses) / len(losses)))

    # ATN examples
    corr = 0
    corr_adv = 0
    l2_lst = []
    linf_lst = []
    for batch_idx, (images, labels) in enumerate(train_loader, start=1):

        images = images.to(config['device'])
        images_adv = atn.perturb(images)

        outputs = net(images)
        outputs_adv = net(images_adv)

        for image, image_adv, output, output_adv, label in zip(images, images_adv, outputs, outputs_adv, labels):

            soft_label = F.softmax(output, dim=0).cpu().detach().numpy()
            soft_label_adv = F.softmax(output_adv, dim=0).cpu().detach().numpy()

            label = label.item()
            pred = np.argmax(soft_label)
            pred_adv = np.argmax(soft_label_adv)

            if label == pred:
                corr += 1

            if label == pred_adv:
                corr_adv += 1

            l2_dist = torch.norm(image - image_adv, 2).item()
            linf_dist = torch.norm(image - image_adv, float('inf')).item()

            l2_lst.append(l2_dist)
            linf_lst.append(linf_dist)
        
        a = sum(l2_lst) / len(l2_lst)
        b = sum(linf_lst) / len(linf_lst)
        print('[%5d/%5d] corr:%5d\tcorr_adv:%5d\tavg.l2:%.4f\tavg.linf:%.4f' % (batch_idx, len(train_loader), corr, corr_adv, a, b))



if __name__ == '__main__':
    main()