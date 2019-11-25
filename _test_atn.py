import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src import VGG, get_train_valid_loader
from src_attacks import P_ATN


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--atn_batch_size', default=32, type=int)
    parser.add_argument('--atn_epoch', default=10, type=int)
    parser.add_argument('--atn_sample', default=0.1, type=float)
    parser.add_argument('--atn_epsilon', default=8, type=int)
    parser.add_argument('--atn_weight', default=None, type=str)
    parser.add_argument('--atn_lr', default=1e-3, type=float)
    args = parser.parse_args()

    # settings
    config = dict()
    config['device'] = args.device
    config['atn_batch_size'] = args.atn_batch_size
    config['atn_epoch'] = args.atn_epoch
    config['atn_sample'] = args.atn_sample
    config['atn_epsilon'] = args.atn_epsilon
    config['atn_weight'] = args.atn_weight
    config['atn_lr'] = args.atn_lr
    weight_path = './weights/vgg16_e086_90.62.pth'

    # classification model
    net = VGG('VGG16').to(config['device'])
    state_dict = torch.load(weight_path, map_location=config['device'])
    net.load_state_dict(state_dict)
    net.eval()

    # train dataloader for testing
    atn_train_loader, _ = get_train_valid_loader(batch_size=config['atn_batch_size'], atn=int(config['atn_sample'] * 40000))

    # train ATN (from scratch or not)
    atn = P_ATN(model=net,
                epsilon=config['atn_epsilon']*4/255,
                weight=config['atn_weight'],
                device=config['device'])

    for epoch_idx in range(1, config['atn_epoch'] + 1):
        losses = []
        lossXs = []
        lossYs = []
        l2_lst = []
        for batch_idx, (images, labels) in enumerate(atn_train_loader):
            loss, lossX, lossY, l2_dist = atn.train(images, labels, learning_rate=config['atn_lr'])
            losses.append(loss)
            lossXs.append(lossX)
            lossYs.append(lossY)
            l2_lst.append(l2_dist)
        avg_loss = sum(losses) / len(losses)
        avg_lossX = sum(lossXs) / len(lossXs)
        avg_lossY = sum(lossYs) / len(lossYs)
        avg_l2 = sum(l2_lst) / len(l2_lst)
        print('[%3d / %3d] Avg.Loss: %.4f(%.4f, %.4f)\tAvg.L2-dist: %.4f' % (epoch_idx, config['atn_epoch'], avg_loss, avg_lossX, avg_lossY, avg_l2))

    # ATN examples
    corr = 0
    corr_adv = 0
    l2_lst = []
    linf_lst = []
    for batch_idx, (images, labels) in enumerate(atn_train_loader, start=1):

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
    print('[%5d/%5d] corr:%5d\tcorr_adv:%5d\tavg.l2:%.4f\tavg.linf:%.4f' % (batch_idx, len(atn_train_loader), corr, corr_adv, a, b))


if __name__ == '__main__':
    main()
