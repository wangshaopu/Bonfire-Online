import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset

from .feature_dataset import *


def __dataset_sample(args, train_set):
    datasizes = [int(len(trainset) * args.percent) for trainset in train_set]
    trainsets = [torch.utils.data.Subset(trainset, torch.randperm(len(trainset))[:datasize]) for trainset, datasize in zip(train_set, datasizes)]

    return trainsets

def global_trainset(args, datasets):
    r"""
    从给定的数据集中抽取一个微型全局训练集
    """
    whole_trainset = ConcatDataset(datasets)
    perm = torch.randperm(len(whole_trainset))
    whole_trainset = Subset(whole_trainset, perm[:args.global_samples] )

    return whole_trainset

def digit(args):
    
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    split_list = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST_M"]
    channeles = [1, 3, 1, 3, 3]
    transform_list = [transform_mnist, transform_svhn, transform_usps, transform_synth, transform_mnistm]

    # 这里的percent是遗留问题
    trainsets = [DigitsDataset(data_path=os.path.join(args.datapath, split), channels=channeles[i], percent=args.percent, train=True,  transform=transform_list[i]) for i, split in enumerate(split_list)]
    valsets = [DigitsDataset(data_path=os.path.join(args.datapath, split), channels=channeles[i], percent=args.percent, train=False, transform=transform_list[i]) for i, split in enumerate(split_list)]

    return trainsets, valsets

def domain(args):
    data_base_path = args.datapath
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])
    
    split_list = ['clipart', 'infograph', 'painting', 'quickdraw','real','sketch']
    trainsets = [DomainNetDataset(data_base_path, split, transform=transform_train) for split in split_list]
    valsets = [DomainNetDataset(data_base_path, split, transform=transform_test, train=False) for split in split_list]

    return __dataset_sample(args, trainsets), valsets


def office(args):
    data_base_path = args.datapath
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])
    
    split_list = ['amazon', 'caltech', 'dslr', 'webcam']
    trainsets = [OfficeDataset(data_base_path, split, transform=transform_office) for split in split_list]
    valsets = [OfficeDataset(data_base_path, split, transform=transform_test, train=False) for split in split_list]

    return __dataset_sample(args, trainsets), valsets
