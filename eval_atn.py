import os
import time
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from src import VGG, get_test_loader
from src import AverageMeter, ProgressMeter, accuracy, write_log
from src_attacks import P_ATN


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = True


def test_accuracy(test_loader, net, config, attack=None):

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for _, (images, labels) in tqdm(enumerate(test_loader, start=1)):

        images = images.to(device=config['device'])
        labels = labels.to(device=config['device'])

        if attack is not None:
            images = attack.perturb(images)

        output = net(images)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

    log_str  = '* Test Accuracy:\t'
    log_str += '\tAcc@1 {top1.avg:6.2f}         '.format(top1=top1)
    log_str += '\tAcc@5 {top5.avg:6.2f}         '.format(top5=top5)
    log_str += '\n'
    print(log_str)

    return top1.avg


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--weight', default=None, type=str)
    parser.add_argument('--atn_sample', default=0.1, type=float)
    parser.add_argument('--atn_epoch', default=10, type=int)
    parser.add_argument('--atn_batch_size', default=32, type=int)
    parser.add_argument('--atn_weight', default=None, type=str)
    parser.add_argument('--atn_lr', default=1e-4, type=float)
    parser.add_argument('--atn_epsilon', default=8, type=int)
    args = parser.parse_args()

    config = dict()
    config['device'] = args.device
    config['weight'] = args.weight
    config['atn_sample'] = args.atn_sample
    config['atn_epoch'] = args.atn_epoch
    config['atn_batch_size'] = args.atn_batch_size
    config['atn_weight'] = args.atn_weight
    config['atn_lr'] = args.atn_lr
    config['atn_epsilon'] = args.atn_epsilon

    # CIFAR-10 dataset (10000)
    test_loader = get_test_loader(batch_size=32)

    # classification network
    net = VGG('VGG16').to(device=config['device'])

    weights = sorted(os.listdir(config['weight']))
    acc1_list = []
    for weight in weights:
        if '.pth' not in weight:
            continue
        print(weight)
        weight_path = os.path.join(config['weight'], weight)
        state_dict = torch.load(weight_path, map_location=config['device'])
        net.load_state_dict(state_dict)
        net.eval()

        # train ATN
        test_loader = get_test_loader(batch_size=config['atn_batch_size'])
        atn = P_ATN(model=net,
                    epsilon=config['atn_epsilon']*4/255,
                    weight=config['atn_weight'],
                    device=config['device'])

        for epoch_idx_atn in range(1, config['atn_epoch'] + 1):
            losses = []
            lossXs = []
            lossYs = []
            l2_lst = []
            for batch_idx, (images, labels) in enumerate(test_loader):
                if batch_idx == int(len(test_loader) * config['atn_sample']):
                    break
                loss, lossX, lossY, l2_dist = atn.train(images, labels, learning_rate=config['atn_lr'])
                losses.append(loss)
                lossXs.append(lossX)
                lossYs.append(lossY)
                l2_lst.append(l2_dist)
            avg_loss = sum(losses) / len(losses)
            avg_lossX = sum(lossXs) / len(lossXs)
            avg_lossY = sum(lossYs) / len(lossYs)
            avg_l2 = sum(l2_lst) / len(l2_lst)
            print('[%3d / %3d] Avg.Loss: %.4f(%.4f, %.4f)\tAvg.L2-dist: %.4f' % (epoch_idx_atn, config['atn_epoch'], avg_loss, avg_lossX, avg_lossY, avg_l2))

        acc1 = test_accuracy(test_loader, net, config, attack=atn)
        acc1_list.append(acc1.cpu().item())

    print(acc1_list)


if __name__ == '__main__':
    main()
