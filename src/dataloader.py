import random
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


IND2CLASS = ('plane', 'car', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck')


def get_test_loader(batch_size):

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
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)
    return test_dataloader


def get_train_valid_loader(batch_size=32, atn=0):

    indices = list(range(50000))
    if atn:
        train_sampler = torch.utils.data.SubsetRandomSampler(random.sample(indices[:40000], atn))
    else:
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
