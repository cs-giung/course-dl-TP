import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from src import VGG, get_train_valid_loader
from src import AverageMeter, ProgressMeter, accuracy, write_log
from src_attacks import PGD_Linf, PGD_L2, FGSM


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = True


def train(train_loader, net, criterion, log_file,
          optimizer, epoch, PGD=None, config=None):

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
    for batch_idx, (images, labels) in enumerate(train_loader, start=1):

        images = images.to(device=config['device'])
        labels = labels.to(device=config['device'])

        if config['pgd_type'] in ['l2', 'linf']:

            if config['pgd_label'] == 0:
                images_adv = PGD.perturb(images, labels)
            else:
                pred_labels = net(images).max(1, keepdim=True)[1].squeeze_()
                images_adv = PGD.perturb(images, pred_labels)
            outputs = net(images_adv)
            loss = criterion(outputs, labels)

        elif config['pgd_type'] == 'fgsm':

            if config['pgd_label'] == 0:
                images_adv = PGD.perturb(images, labels)
            else:
                pred_labels = net(images).max(1, keepdim=True)[1].squeeze_()
                images_adv = PGD.perturb(images, pred_labels)
            outputs = net(images)
            outputs_adv = net(images_adv)
            loss = criterion(outputs, labels)
            loss_adv = criterion(outputs_adv, labels)
            loss = 0.5 * loss + 0.5 * loss_adv

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
    parser.add_argument('--pgd_type', default=None, type=str)
    parser.add_argument('--pgd_epsilon', default=8, type=int)
    parser.add_argument('--pgd_label', default=0, type=int)
    args = parser.parse_args()

    config = dict()
    config['device'] = args.device
    config['num_epoch'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['lr_decay'] = args.lr_decay
    config['pgd_type'] = args.pgd_type
    config['pgd_epsilon'] = args.pgd_epsilon
    config['pgd_label'] = args.pgd_label

    # CIFAR-10 dataset (40000 + 10000)
    train_loader, valid_loader = get_train_valid_loader(batch_size=config['batch_size'])

    # classification network
    net = VGG('VGG16').to(device=config['device'])

    # train settings
    learning_rate = config['learning_rate']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    best_valid_acc1 = 0

    output_path = './train_pgd_{:%Y-%m-%d-%H-%M-%S}/'.format(datetime.now())
    log_file = output_path + 'train_log.txt'
    os.mkdir(output_path)

    for epoch_idx in range(1, config['num_epoch'] + 1):

        # learning rate scheduling
        if epoch_idx % config['lr_decay'] == 0:
            learning_rate *= 0.5
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        # train & valid
        if config['pgd_type'] == 'l2':
            PGD = PGD_L2(model=net, epsilon=config['pgd_epsilon']*4/255)
            _ = train(train_loader, net, criterion, log_file, optimizer, epoch_idx, PGD=PGD, config=config)
        elif config['pgd_type'] == 'linf':
            PGD = PGD_Linf(model=net, epsilon=config['pgd_epsilon']*4/255)
            _ = train(train_loader, net, criterion, log_file, optimizer, epoch_idx, PGD=PGD, config=config)
        elif config['pgd_type'] == 'fgms':
            FGSM = FGSM(model=net, num_steps=config['pgd_epsilon'])
            _ = train(train_loader, net, criterion, log_file, optimizer, epoch_idx, PGD=FGSM, config=config)
        else:
            _ = train(train_loader, net, criterion, log_file, optimizer, epoch_idx, PGD=None, config=config)
        valid_acc1 = valid(valid_loader, net, criterion, log_file, config=config)

        # save best
        if valid_acc1 > best_valid_acc1:
            best_valid_acc1 = valid_acc1
            file_name = output_path + 'vgg16_e%03d_%.2f.pth' % (epoch_idx, best_valid_acc1)
            torch.save(net.state_dict(), file_name)
            print('epoch=%003d, acc=%.4f saved.\n' % (epoch_idx, best_valid_acc1))


if __name__ == '__main__':
    main()
