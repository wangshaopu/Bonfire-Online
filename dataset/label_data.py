import torch
import torchvision.transforms as transforms
from torchvision import datasets

from .util import data_split, dirichlet


def __dataset_split(args, train_set, val_set):
    r"""
    按照狄利克雷分布划分，并放入loader，并返回一个全局的测试集
    """
    label_distribution = dirichlet(args)

    client_idcs = data_split(args, train_set.targets, label_distribution)
    train_sets = [torch.utils.data.Subset(
        train_set, client_idcs[rank]) for rank in range(args.nprocs)]
    
    val_idcs = data_split(args, val_set.targets, label_distribution)
    val_sets = [torch.utils.data.Subset(
        val_set, val_idcs[rank]) for rank in range(args.nprocs)]

    return train_sets, val_sets, val_set

def cifar10(args):
    dataset = datasets.CIFAR10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))
    train_dataset = dataset(
        args.datapath, download=False, train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = dataset(
        args.datapath, download=False, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return __dataset_split(args, train_dataset, val_dataset)

def cifar100(args):
    dataset = datasets.CIFAR100
    normalize = transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762))

    train_dataset = dataset(
        args.datapath, download=False, train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = dataset(
        args.datapath, download=False, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return __dataset_split(args, train_dataset, val_dataset)

def mnist(args):
    dataset = datasets.MNIST
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    train_dataset = dataset(args.datapath, download=False, train=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                            ]))

    val_dataset = dataset(args.datapath, download=False, train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize,
                          ]))
    return __dataset_split(args, train_dataset, val_dataset)


dataset_dict = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'mnist': mnist,
}