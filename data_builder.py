import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from data.autoaugment import CIFAR10Policy, Cutout
import torch

####################################################
# data loader                                      #
#                                                  #
####################################################

def build_data(dpath: str = None, batch_size=36, cutout=False, workers=1, use_cifar10=True, auto_aug=False,
               dataset='CIFAR10', train_val_split=True, imagenet_train_dir=None, imagenet_val_dir=None):

    if dataset == 'CIFAR10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'CIFAR100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
    elif dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    elif dataset == 'fashion-mnist':
        mean = (0.2860,)
        std = (0.3530,)
    else:
        assert False, "Unknown dataset : {dataset}"
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())
    aug.append(transforms.ToTensor())
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    test_dataset = None
    val_dataset = None

    if (imagenet_train_dir is None or imagenet_val_dir is None) and dpath is None:
        assert False, "Please input your dataset dir path via --dataset_path [dataset_dir] or " \
                      "--train_dir [imagenet_train_dir] --val_dir [imagenet_train_dir]"

    if use_cifar10:
        transform_train = transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                         ])
        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])
        train_dataset = CIFAR10(root=dpath, train=True, download=True, transform=transform_train)
        if train_val_split:
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, lengths=[40000, 10000],
                                                                  generator=torch.Generator().manual_seed(42)
                                                                  )
            test_dataset = CIFAR10(root=dpath, train=False, download=True, transform=transform_test)
        else:
            val_dataset = CIFAR10(root=dpath, train=False, download=True, transform=transform_test)

    elif dataset == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


        train_dataset = ImageFolder(root=imagenet_train_dir,
                                    transform=transform_train)
        if train_val_split:
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, lengths=[40000, 10000],#if you need split by other means, you have to recalculate the precise split numbers
                                                                  generator=torch.Generator().manual_seed(42)
                                                                  )
            test_dataset = ImageFolder(root=imagenet_val_dir, transform=transform_test)
        else:
            val_dataset = ImageFolder(root=imagenet_val_dir, transform=transform_test)

    elif dataset == 'fashion-mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = datasets.FashionMNIST(root=dpath,
                                              train=True, transform=transform_train, download=True)
        val_dataset = datasets.FashionMNIST(root=dpath,
                                            train=False,
                                            transform=transform_test,
                                            download=True)

    elif dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])

        train_dataset = datasets.MNIST(root=dpath,
                                              train=True, transform=transform_train, download=True)
        val_dataset = datasets.MNIST(root=dpath,
                                            train=False,
                                            transform=transform_test,
                                            download=True)

    elif dataset == 'CIFAR100':
        transform_train = transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=mean, std=std)
                         ])
        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)
                        ])

        train_dataset = datasets.CIFAR100(root=dpath, train=True, transform=transform_train, download=True)
        val_dataset = datasets.CIFAR100(root=dpath, train=False, transform=transform_test, download=True)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=dpath, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=dpath, train=False, download=True, transform=transform_test)


    #multi-GPUs for distributed computation
    if torch.cuda.device_count() > 1:
        if train_val_split:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                num_workers=workers,
                drop_last=True,
                pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                drop_last=False,
                pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                sampler=test_sampler,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                drop_last=False,
                pin_memory=True)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                num_workers=workers,
                drop_last=True,
                pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                drop_last=False,
                pin_memory=True)
            test_loader = None
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=workers, pin_memory=True)
        if train_val_split:
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=workers, pin_memory=True)
        else:
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=workers, pin_memory=True)
            test_loader = None
    return train_loader, val_loader, test_loader



