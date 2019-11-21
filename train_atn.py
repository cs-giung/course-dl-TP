import os
import time
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from src import get_train_valid_loader
from src import VGG, AAE_ATN
from src import AverageMeter, ProgressMeter, accuracy, write_log


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = True


def train(train_loader, net, criterion, log_file,
          optimizer, epoch, ATN, config=None):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch)
    )

    end = time.time()
    atn_cnt = 0
    for batch_idx, (images, labels) in enumerate(train_loader, start=1):

        images = images.to(device=config['device'])
        labels = labels.to(device=config['device'])

        images, is_atn = ATN.perturb(images, threshold=config['atn_threshold'])
        if is_atn:
            atn_cnt += 1

        outputs = net(images)
        loss = criterion(outputs, labels)

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.data, images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx == 1 or batch_idx % 400 == 0:
            progress.display(batch_idx)
            write_log(log_file, progress.log_str(batch_idx))

    print(atn_cnt)

    return top1.avg


def valid(valid_loader, net, criterion, log_file, config=None):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        end = time.time()
        for _, (images, labels) in enumerate(valid_loader, start=1):

            images = images.to(device=config['device'])
            labels = labels.to(device=config['device'])

            output = net(images)
            loss = criterion(output, labels)

            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.data, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        log_str  = '* Validation:\t'
        log_str += '\tTime {batch_time.avg:6.3f}         '.format(batch_time=batch_time)
        log_str += '\tLoss {losses.avg:.4e}             '.format(losses=losses)
        log_str += '\tAcc@1 {top1.avg:6.2f}         '.format(top1=top1)
        log_str += '\tAcc@5 {top5.avg:6.2f}         '.format(top5=top5)
        log_str += '\n'
        print(log_str)
        write_log(log_file, log_str)

    return top1.avg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr_decay', default=20, type=int)
    parser.add_argument('--atn_epoch', default=10, type=int)
    parser.add_argument('--atn_sample', default=0.1, type=float)
    parser.add_argument('--atn_alpha', default=0.1, type=float)
    parser.add_argument('--atn_beta', default=0.7, type=float)
    parser.add_argument('--atn_scratch', default=0, type=int)
    parser.add_argument('--atn_threshold', default=8, type=int)
    parser.add_argument('--atn_debug', default=0, type=int)
    args = parser.parse_args()

    config = dict()
    config['device'] = args.device
    config['num_epoch'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['lr_decay'] = args.lr_decay
    config['atn_epoch'] = args.atn_epoch
    config['atn_sample'] = args.atn_sample
    config['atn_alpha'] = args.atn_alpha
    config['atn_beta'] = args.atn_beta
    config['atn_scratch'] = args.atn_scratch
    config['atn_threshold'] = args.atn_threshold
    config['atn_debug'] = args.atn_debug

    # CIFAR-10 dataset (40000 + 10000)
    train_loader, valid_loader = get_train_valid_loader(batch_size=config['batch_size'])

    # classification network
    net = VGG('VGG16').to(device=config['device'])

    # train settings
    learning_rate = config['learning_rate']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    best_valid_acc1 = 0

    output_path = './train_atn_{:%Y-%m-%d-%H-%M-%S}/'.format(datetime.now())
    log_file = output_path + 'train_log.txt'
    os.mkdir(output_path)

    for epoch_idx in range(1, config['num_epoch'] + 1):

        # learning rate scheduling
        if epoch_idx % config['lr_decay'] == 0:
            learning_rate *= 0.5
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        # train ATN
        atn_train_loader, _ = get_train_valid_loader(batch_size=config['batch_size'], atn=int(config['atn_sample']*40000))
        if config['atn_scratch']:
            atn = AAE_ATN(device=config['device'],
                        target_classifier=net)
            lr = 1e-3
        else:
            atn = AAE_ATN(device=config['device'],
                        weight='./weights/base_atn_conv.pth',
                        target_classifier=net)
            lr = 1e-4

        for epoch_idx_atn in range(1, config['atn_epoch'] + 1):
            losses = []
            lossXs = []
            lossYs = []
            l2_lst = []
            for batch_idx, (images, labels) in enumerate(atn_train_loader):
                loss,  lossX, lossY, l2_dist = atn.train(images, alpha=config['atn_alpha'], beta=config['atn_beta'], learning_rate=lr)
                losses.append(loss)
                lossXs.append(lossX)
                lossYs.append(lossY)
                l2_lst.append(l2_dist)
            avg_loss = sum(losses) / len(losses)
            avg_lossX = sum(lossXs) / len(lossXs)
            avg_lossY = sum(lossYs) / len(lossYs)
            avg_l2 = sum(l2_lst) / len(l2_lst)
            print('[%3d / %3d] Avg.Loss: %.4f(%.4f, %.4f)\tAvg.L2-dist: %.4f' % (epoch_idx_atn, config['atn_epoch'], avg_loss, avg_lossX, avg_lossY, avg_l2))

        # DEBUG
        if config['atn_debug']:
            corr = 0
            corr_adv = 0
            l2_lst = []
            linf_lst = []
            for batch_idx, (images, labels) in enumerate(valid_loader, start=1):

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
            print('[%5d/%5d] corr:%5d\tcorr_adv:%5d\tavg.l2:%.4f\tavg.linf:%.4f' % (batch_idx, len(valid_loader), corr, corr_adv, a, b))

        # train & valid
        _ = train(train_loader, net, criterion, log_file, optimizer, epoch_idx, ATN=atn, config=config)
        valid_acc1 = valid(valid_loader, net, criterion, log_file, config=config)

        # save best
        if valid_acc1 > best_valid_acc1:
            best_valid_acc1 = valid_acc1
            file_name = output_path + 'vgg16_e%03d_%.2f.pth' % (epoch_idx, best_valid_acc1)
            torch.save(net.state_dict(), file_name)
            print('epoch=%003d, acc=%.4f saved.\n' % (epoch_idx, best_valid_acc1))


if __name__ == '__main__':
    main()
