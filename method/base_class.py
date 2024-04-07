import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.modules.batchnorm import _BatchNorm

import dataset.feature_data as feature_data
import dataset.label_data as label_data
import nets

from .tools import *

# 分类不平衡数据
label_dataset = {"cifar10": label_data.cifar10, "cifar100": label_data.cifar100, "mnist": label_data.mnist}
# 特征不平衡数据
feature_dataset = {"digit": feature_data.digit, "office": feature_data.office, "domain": feature_data.domain}

num_classes = {'cifar100': 100, 'femnist': 62, 'mixed_femnist': 62, 'celeba': 2, 'shakespeare': 80, 'agnews': 4,}

class Server(object):
    def __init__(self, args) -> None:
        seed_everything(args.seed)
        args.num_classes = num_classes[args.dataset] if args.dataset in num_classes.keys() else 10
        args.criterion = nn.CrossEntropyLoss().to(args.device)

        self.criterion = args.criterion
        self.num_classes = args.num_classes
        self.size = args.nprocs  # 机器数量
        self.client_list = None
        self.device = args.device
        self.datapath = args.datapath
        # 事实上，大部分数据集都是10分类
        
        if args.dataset in label_dataset.keys():
            self.train_sets, self.val_sets, self.global_val_set = label_dataset[args.dataset](args)
            # 主动切分时考虑增加一个全局测试集
            self.global_val_loader = dataloader(args, self.global_val_set, False)
        elif args.dataset in feature_dataset.keys():
            # 特征不平衡则需要覆盖客户端数量
            args.datapath = os.path.join(args.datapath, 'feature')
            self.train_sets, self.val_sets = feature_dataset[args.dataset](args)
            self.size = len(self.train_sets) 
        else:
            raise ValueError("Dataset not supported")
    
        self.global_model = getattr(nets, args.arch)(
            num_classes=self.num_classes).to(self.device)

    @staticmethod
    def add_arguements(parser: ArgumentParser):
        r"""
        添加超参数
        """
        print('No arguements added.')

    @torch.no_grad()
    def aggregate(self):
        r"""
        聚合客户端模型参数到全局模型
        """
        for param in self.global_model.parameters():
            param.data.zero_()

        # 一个常见的错误，即忽略BN层
        for module in self.global_model.modules():
            if isinstance(module, _BatchNorm):
                module.running_mean.zero_()
                module.running_var.zero_()
        for client in self.client_list:
            c_model = client.model
            for s_param, c_param in zip(self.global_model.parameters(), c_model.parameters()):
                s_param.data += c_param.data.clone() / self.size

            # 聚合统计特征
            for s_module, c_module in zip(self.global_model.modules(), c_model.modules()):
                if isinstance(c_module, _BatchNorm):
                    s_module.running_mean += c_module.running_mean.clone() / self.size
                    s_module.running_var += c_module.running_var.clone() / self.size       

    @torch.no_grad()
    def download_models(self):
        r"""
        分发全局模型参数到各个客户端
        """
        for client in self.client_list:
            c_model = client.model
            for s_param, c_param in zip(self.global_model.parameters(), c_model.parameters()):
                c_param.data.copy_(s_param.data)

            # 顺便同步一下bn层的mean和var
            for s_module, c_module in zip(self.global_model.modules(), c_model.modules()):
                if isinstance(s_module, _BatchNorm):
                    c_module.running_mean.copy_(s_module.running_mean)
                    c_module.running_var.copy_(s_module.running_var)
    
    @torch.no_grad()
    def global_validate(self):
        r"""
        返回全局模型在全局数据集上的精度
        """
        return [validate(self.global_model, self.global_val_loader, self.criterion, self.device)]

    @torch.no_grad()
    def personlized_validate(self, per_val=False):
        if per_val:
            # 多对多
            return [client.validate() for client in self.client_list]
        else:
            # 多对一
            return [client.validate(self.global_val_loader) for client in self.client_list]

class Client(object):
    def __init__(self, args, train_set, val_set) -> None:
        self.device = args.device
        self.criterion = args.criterion
        self.train_loader = dataloader(args, train_set, True)
        self.val_loader = dataloader(args, val_set, False)
        
        self.model = getattr(nets, args.arch)(
            num_classes=args.num_classes).to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)

        self.epochs = args.epochs
        self.scaler = GradScaler() # 用于混合精度训练

    def validate(self, data_loader=None):
        if data_loader is None:
            data_loader = self.val_loader

        return validate(self.model, data_loader, self.criterion, self.device)

def train(epochs, model, optimizer, train_loader, loss_fun, scaler, device):
    model.train()

    for _ in range(epochs):
        for img, target in train_loader:
            img, target = img.to(device, non_blocking=True), target.to(
                device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(img)
                loss = loss_fun(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

def validate(model, val_loader, loss_fun, device):
    model.eval()
    # test_loss = 0
    correct = 0

    for data, target in val_loader:
        data = data.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).long()

        output = model(data)
        
        # test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    
    # test_loss/len(val_loader)
    return correct / len(val_loader.dataset) * 100