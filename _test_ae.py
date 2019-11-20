import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from src import get_train_valid_loader
from src import get_test_loader
from src import AverageMeter, ProgressMeter
from ae_cifar10.ae2 import AutoEncoder


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = True


def train(train_loader, net, criterion,
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

    return losses.avg


def valid(valid_loader, net, criterion, config=None):

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

    train_loader, valid_loader = get_train_valid_loader(batch_size=32)

    net = AutoEncoder().to(config['device'])
    optimizer = optim.Adam(net.parameters())
    criterion = nn.MSELoss()

    for epoch_idx in range(1, config['num_epoch'] + 1):
        train_loss = train(train_loader, net, criterion, optimizer, epoch_idx, config)
        valid_loss = valid(valid_loader, net, criterion, config)

    test_loader = get_test_loader(batch_size=1)
    l2_lst = []
    linf_lst = []
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(config['device'])
        outputs = net(images)
        l2_lst.append(torch.norm(images - outputs, 2).item())
        linf_lst.append(torch.norm(images - outputs, float('inf')).item())
        if batch_idx % 1000 == 0:
            avg_l2 = sum(l2_lst) / len(l2_lst)
            avg_linf = sum(linf_lst) / len(linf_lst)
            print('[%5d] l2: %.4f\tlinf:%.4f' % (batch_idx, avg_l2, avg_linf))
            l2_lst = []; linf_lst = []


if __name__ == '__main__':
    main()
