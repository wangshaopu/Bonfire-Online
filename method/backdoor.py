import torch

from .base_class import Client, Server, train
from .fedavg import FedAvgClient
from .tools import weight_flatten


def pairwise_distance(client_list):
    r"""
    获取所有客户端两两之间的距离
    """
    distance = torch.zeros([len(client_list), len(client_list)])

    for i, client_i in enumerate(client_list):
        for j, client_j in enumerate(client_list):
            param_1 = weight_flatten(client_i.model)
            param_2 = weight_flatten(client_j.model)
            distance[i][j] = (param_1 - param_2).square().sum()

    return distance

def krum(honest_num, client_list, global_model):
    r"""
    找到一个和所有参与方都接近的客户端作为全局模型
    """
    distance = pairwise_distance(client_list)
    score = []
    for d in distance:
        score.append(d.sort()[0][:honest_num].sum())

    target = score.index(min(score))
    global_model.load_state_dict(client_list[target].model.state_dict())


class Backdoor(Server):
    def __init__(self, args):
        super().__init__(args)
        self.attack = args.attack
        self.defend = args.defend
        self.malicious = args.malicious

        # 选定恶意和正常用户数量
        self.malicious, self.benign = self.pick_clients()

        # 准备客户端，前边是attack用户，后边是fedavg
        self.client_list = [attack_dict[self.attack](args, self.train_sets[rank], self.val_sets[rank]) for rank in range(self.malicious)] + [FedAvgClient(args, self.train_sets[rank], self.val_sets[rank]) for rank in range(self.benign)]

        self.download_models()
        
    @staticmethod
    def add_arguements(parser):
        parser.add_argument('--attack', type=str, default='noise', help='攻击方法')
        parser.add_argument('--defend', type=str, default='krum', help='防御方法')
        parser.add_argument('--malicious', type=float, default='0.6', help='恶意客户端占比')

    def aggregate(self):
        r"""
        防御一般发生在聚合阶段
        """
        defend_dict[self.defend](self.benign, self.client_list, self.global_model)

    def pick_clients(self):
        r"""
        根据malicious计算恶意和正常用户数量
        """
        malicious_num = int(self.size * self.malicious)
        benign_num = self.size - malicious_num

        return malicious_num, benign_num

    def train(self, epoch):
        for client in self.client_list:
            client.train()

        self.aggregate()
        self.download_models()

class NoiseClient(Client):
    r"""
    以噪声标签为例
    """
    def __init__(self, args, train_set, val_set) -> None:
        super().__init__(args, train_set, val_set)
        self.num_classes = args.num_classes

    def train(self):
        self.model.train()

        for _ in range(self.epochs):
            for img, target in self.train_loader:
                img = img.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                target = (target + 3) % self.num_classes # 标签偏移带来的噪声

                self.optimizer.zero_grad()
                # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    output = self.model(img)
                    loss = self.criterion(output, target)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

attack_dict = {
    'noise': NoiseClient
}

defend_dict = {
    'krum': krum
}