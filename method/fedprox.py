import torch
from torch.linalg import norm

from .base_class import Client, Server
from .tools import weight_flatten


class FedProx(Server):
    def __init__(self, args):
        super().__init__(args)
        self.client_list = [FedProxClient(args, self.train_sets[rank], self.val_sets[rank]) for rank in range(self.size)]

        self.download_models()

    @staticmethod
    def add_arguements(parser):
        parser.add_argument('--mu', type=float, default=0.01, help='mu for FedProx')

    def train(self, epoch):
        for client in self.client_list:
            client.train(self.global_model)

        self.aggregate()
        self.download_models()


class FedProxClient(Client):
    def __init__(self, args, train_set, val_set):
        super().__init__(args, train_set, val_set)
        self.mu = args.mu

    def train(self, global_model):
        src_model = weight_flatten(global_model)
        self.model.train()

        for _ in range(self.epochs):
            for img, target in self.train_loader:
                img, target = img.to(self.device, non_blocking=True), target.to(
                    self.device, non_blocking=True)

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = self.model(img)
                    loss = self.criterion(output, target)

                    # FedProx
                    flatten_model = weight_flatten(self.model)
                    sub = src_model - flatten_model
                    loss += self.mu / 2 * torch.square(sub).sum()

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()