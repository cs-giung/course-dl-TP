import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src import get_test_loader, get_train_valid_loader
from src import VGG, AAE_ATN


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--atn_epoch', default=10, type=int)
    parser.add_argument('--atn_sample', default=0.1, type=float)
    parser.add_argument('--atn_scratch', default=0, type=int)
    parser.add_argument('--atn_alpha', default=0.5, type=float)
    parser.add_argument('--atn_beta', default=0.99, type=float)
    args = parser.parse_args()

    # settings
    config = dict()
    config['device'] = args.device
    config['num_epoch'] = args.epochs
    config['batch_size'] = args.batch_size
    config['atn_epoch'] = args.atn_epoch
    config['atn_sample'] = args.atn_sample
    config['atn_scratch'] = args.atn_scratch
    config['atn_alpha'] = args.atn_alpha
    config['atn_beta'] = args.atn_beta
    weight_path = './weights/vgg16_e086_90.62.pth'

    # classification model
    net = VGG('VGG16').to(config['device'])
    state_dict = torch.load(weight_path, map_location=config['device'])
    net.load_state_dict(state_dict)
    net.eval()

    # test dataset
    loader, _ = get_train_valid_loader(batch_size=32)
    # loader = get_test_loader(batch_size=32)

    # train ATN
    if config['atn_scratch']:
        atn = AAE_ATN(device=config['device'],
                      target_classifier=net)
        lr = 1e-3
    else:
        atn = AAE_ATN(device=config['device'],
                      weight='./weights/base_atn_conv.pth',
                      target_classifier=net)
        lr = 1e-4

    for epoch_idx in range(1, config['atn_epoch'] + 1):
        losses = []
        l2_lst = []
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx == int(config['atn_sample'] * len(loader)):
                break
            loss, l2_dist = atn.train(images, labels, beta=config['atn_beta'], learning_rate=lr)
            losses.append(loss)
            l2_lst.append(l2_dist)
        avg_loss = sum(losses) / len(losses)
        avg_l2 = sum(l2_lst) / len(l2_lst)
        print('[%3d / %3d] Avg.Loss: %f\tAvg.L2-dist: %f' % (epoch_idx, config['atn_epoch'], avg_loss, avg_l2))

    # ATN examples
    corr = 0
    corr_adv = 0
    l2_lst = []
    linf_lst = []
    for batch_idx, (images, labels) in enumerate(loader, start=1):

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
    print('[%5d/%5d] corr:%5d\tcorr_adv:%5d\tavg.l2:%.4f\tavg.linf:%.4f' % (batch_idx, len(loader), corr, corr_adv, a, b))


if __name__ == '__main__':
    main()
