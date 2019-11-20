"""
train ATN as identity.

"""


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

from src import _atn_conv
from src import AverageMeter, ProgressMeter, accuracy, write_log


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
          optimizer, epoch, config=None):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    end = time.time()
    for batch_idx, (images, _) in enumerate(train_loader, start=1):

        images = images.to(device=config['device'])

        outputs = net(images)
        loss = criterion(outputs, images)

        losses.update(loss.data, images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx == 1 or batch_idx % 400 == 0:
            progress.display(batch_idx)
            write_log(log_file, progress.log_str(batch_idx))

    return True


def valid(valid_loader, net, criterion, log_file, config=None):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    with torch.no_grad():
        end = time.time()
        for _, (images, _) in enumerate(valid_loader, start=1):

            images = images.to(device=config['device'])

            output = net(images)
            loss = criterion(output, images)

            losses.update(loss.data, images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        log_str  = '* Validation:\t'
        log_str += '\tTime {batch_time.avg:6.3f}         '.format(batch_time=batch_time)
        log_str += '\tLoss {losses.avg:.4e}             '.format(losses=losses)
        log_str += '\n'
        print(log_str)
        write_log(log_file, log_str)

    return losses.avg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    config = dict()
    config['device'] = args.device
    config['num_epoch'] = args.epochs
    config['batch_size'] = args.batch_size

    # CIFAR-10 dataset (40000 + 10000)
    train_loader, valid_loader = get_train_valid_loader(batch_size=config['batch_size'])

    # ATN network
    net = _atn_conv().to(device=config['device'])

    # train settings
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    best_valid_loss = 1000

    output_path = './pretrain_atn_{:%Y-%m-%d-%H-%M-%S}/'.format(datetime.now())
    log_file = output_path + 'train_log.txt'
    os.mkdir(output_path)

    for epoch_idx in range(1, config['num_epoch'] + 1):

        # train & valid
        _ = train(train_loader, net, criterion, log_file, optimizer, epoch_idx, config=config)
        valid_loss = valid(valid_loader, net, criterion, log_file, config=config)

        # save best
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            file_name = output_path + 'vgg16_e%03d_%.4f.pth' % (epoch_idx, best_valid_loss)
            torch.save(net.state_dict(), file_name)
            print('epoch=%003d, loss=%.4f saved.\n' % (epoch_idx, best_valid_loss))


if __name__ == '__main__':
    main()
