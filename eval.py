import os
import time
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src import VGG
from src import PGD_Linf, FGSM, PGD_L2
from src import AverageMeter, ProgressMeter, accuracy, write_log


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = True


def get_test_loader(batch_size=32):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.500, 0.500, 0.500),
            (0.250, 0.250, 0.250)
        )
    ])

    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=test_transform
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
    )

    return test_dataloader


def test_accuracy(test_loader, net, config, attack=None):

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for _, (images, labels) in tqdm(enumerate(test_loader, start=1)):

        images = images.to(device=config['device'])
        labels = labels.to(device=config['device'])

        if attack is not None:
            images = attack.perturb(images, labels)

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
    parser.add_argument('--attack', default=None, type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    args = parser.parse_args()

    if args.attack not in [None, 'fgsm', 'l2_pgd', 'linf_pgd']:
        print('--attack [fgsm or l2_pgd or linf_pgd]')
        exit()

    config = dict()
    config['device'] = args.device
    config['weight'] = args.weight
    config['attack'] = args.attack
    config['epsilon'] = args.epsilon

    # CIFAR-10 dataset (10000)
    test_loader = get_test_loader()

    # classification network
    net = VGG('VGG16').to(device=config['device'])
    if '.pth' in config['weight']:
        print(config['weight'])
        state_dict = torch.load(config['weight'], map_location=config['device'])
        net.load_state_dict(state_dict)
        net.eval()

        # test
        if config['attack'] == None:
            _ = test_accuracy(test_loader, net, config, attack=None)
        elif config['attack'] == 'fgsm':
            attack_FGSM = FGSM(model=net, num_steps=config['epsilon'])
            _ = test_accuracy(test_loader, net, config, attack=attack_FGSM)
        elif config['attack'] == 'linf_pgd':
            attack_PGD_Linf = PGD_Linf(model=net, epsilon=config['epsilon']*4/255)
            _ = test_accuracy(test_loader, net, config, attack=attack_PGD_Linf)
        elif config['attack'] == 'l2_pgd':
            attack_PGD_L2 = PGD_L2(model=net, epsilon=config['epsilon']*4/255)
            _ = test_accuracy(test_loader, net, config, attack=attack_PGD_L2)

    else:
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

            # test
            if config['attack'] == None:
                acc1 = test_accuracy(test_loader, net, config, attack=None)
            elif config['attack'] == 'fgsm':
                attack_FGSM = FGSM(model=net, num_steps=config['epsilon'])
                acc1 = test_accuracy(test_loader, net, config, attack=attack_FGSM)
            elif config['attack'] == 'linf_pgd':
                attack_PGD_Linf = PGD_Linf(model=net, epsilon=config['epsilon']*4/255)
                acc1 = test_accuracy(test_loader, net, config, attack=attack_PGD_Linf)
            elif config['attack'] == 'l2_pgd':
                attack_PGD_L2 = PGD_L2(model=net, epsilon=config['epsilon']*4/255)
                acc1 = test_accuracy(test_loader, net, config, attack=attack_PGD_L2)

            acc1_list.append(acc1.cpu().item())
        print(acc1_list)


if __name__ == '__main__':
    main()
