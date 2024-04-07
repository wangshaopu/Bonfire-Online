import torch
from torch.nn.modules.batchnorm import _BatchNorm

from .base_class import Client, Server
from .fedavg import FedAvgClient


class FedBN(Server):
    def __init__(self, args):
        super().__init__(args)
        self.client_list = [FedAvgClient(args, self.train_sets[rank], self.val_sets[rank]) for rank in range(self.size)]

        self.download_models()

    @torch.no_grad()
    def download_models(self):
        r"""
        仅覆写下载部分，不下载BN层参数
        """
        for client in self.client_list:
            c_model = client.model
            for (name, s_param), c_param in zip(self.global_model.named_parameters(), c_model.parameters()):
                if 'bn' not in name:
                    c_param.data.copy_(s_param.data)

    def train(self, epoch):
        for client in self.client_list:
            client.train()

        self.aggregate()
        self.download_models()