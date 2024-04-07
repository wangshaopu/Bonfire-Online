import torch

from .base_class import Client, Server, train


class FedAvg(Server):
    def __init__(self, args):
        super().__init__(args)
        self.client_list = [FedAvgClient(args, self.train_sets[rank], self.val_sets[rank]) for rank in range(self.size)]

        self.download_models()

    def train(self, epoch):
        for client in self.client_list:
            client.train()

        self.aggregate()
        self.download_models()


class FedAvgClient(Client):
    def train(self):
        train(self.epochs, self.model, self.optimizer, self.train_loader, self.criterion, self.scaler, self.device)
        # self.model.train()

        # for _ in range(self.epochs):
        #     for img, target in self.train_loader:
        #         img, target = img.to(self.device, non_blocking=True), target.to(
        #             self.device, non_blocking=True)

        #         self.optimizer.zero_grad()
        #         with torch.cuda.amp.autocast():
        #             output = self.model(img)
        #             loss = self.criterion(output, target)
        #             self.scaler.scale(loss).backward()
        #             self.scaler.step(self.optimizer)
        #             self.scaler.update()