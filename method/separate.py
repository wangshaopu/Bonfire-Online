from .base_class import Server
from .fedavg import FedAvgClient


class Separate(Server):
    def __init__(self, args):
        super().__init__(args)
        self.client_list = [FedAvgClient(args, self.train_sets[rank], self.val_sets[rank]) for rank in range(self.size)]

        self.download_models()

    def train(self, epoch):
        for client in self.client_list:
            client.train()