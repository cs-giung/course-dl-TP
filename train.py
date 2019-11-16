import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src import VGG
from src import PGD_Linf, PGD_L2
from train_func import AverageMeter, ProgressMeter, accuracy, write_log


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = True


def get_train_valid_loader(batch_size=32):

    indices = list(range(50000))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[:40000])
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[40000:])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.500, 0.500, 0.500),
            (0.250, 0.250, 0.250)
        )
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.500, 0.500, 0.500),
            (0.250, 0.250, 0.250)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=train_transform
    )

    valid_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=valid_transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
    )

    return train_dataloader, valid_dataloader


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

        if PGD is not None:
            if config['pgd_label'] == 0:
                images = PGD.perturb(images, labels)
            else:
                pred_labels = net(images).max(1, keepdim=True)[1].squeeze_()
                images = PGD.perturb(images, pred_labels)

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
    parser.add_argument('--pgd_train', default=None, type=str)
    parser.add_argument('--pgd_epsilon', default=8, type=int)
    parser.add_argument('--pgd_label', default=0, type=int)
    args = parser.parse_args()

    config = dict()
    config['device'] = args.device
    config['num_epoch'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['lr_decay'] = args.lr_decay
    config['pgd_train'] = args.pgd_train
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

    output_path = './train_{:%Y-%m-%d-%H-%M-%S}/'.format(datetime.now())
    log_file = output_path + 'train_log.txt'
    os.mkdir(output_path)

    for epoch_idx in range(1, config['num_epoch'] + 1):

        # learning rate scheduling
        if epoch_idx % config['lr_decay'] == 0:
            learning_rate *= 0.5
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        # train & valid
        if config['pgd_train'] == 'l2':
            PGD = PGD_L2(model=net, epsilon=config['pgd_epsilon']*4/255)
            _ = train(train_loader, net, criterion, log_file, optimizer, epoch_idx, PGD=PGD, config=config)
        elif config['pgd_train'] == 'linf':
            PGD = PGD_Linf(model=net, epsilon=config['pgd_epsilon']*4/255)
            _ = train(train_loader, net, criterion, log_file, optimizer, epoch_idx, PGD=PGD, config=config)
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
