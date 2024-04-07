from argparse import ArgumentParser

import torch

import method


def main(args, server):
    server = server(args)
    for comm in range(args.iters):
        print("============ 通信轮数 {} ============".format(comm))
        server.train(comm)
        acc_list = server.global_validate()
        # acc_list = server.personlized_validate(per_val=True)

        print('Val Acc: ', end='')
        for acc in acc_list:
            print("{:.4f}".format(acc), end=',')
        print()
        print("Averge: {:.4f}".format(sum(acc_list) / len(acc_list)))


if __name__ == '__main__':
    parser = ArgumentParser()
    # 这里仅展示一些通用的参数
    parser.add_argument('--method', type=str, default='Backdoor', help='方法名称', choices=['FedAvg', 'FedProx', 'FedBN', 'Separate', 'Backdoor']) # 是否要开启后门攻防模块
    parser.add_argument('--seed', type = int, default=2526, help ='随机种子')
    parser.add_argument('--arch', default='Softmax', help ='模型类型', choices=["DigitModel", "AlexNet", "Softmax", "FedAvgCNN", "FedAvgCNN_BN"])
    parser.add_argument('--gpu', type = int, default=0, help ='使用的GPU')

    parser.add_argument('--datapath', default=r'/path/to/data', help ='数据集位置')
    parser.add_argument('--dataset', default='mnist', help ='数据集类型', choices=["digit", "domain", "office", "mnist", "cifar10"])
    parser.add_argument('--percent', type = float, default=1.0, help ='训练集占比')

    parser.add_argument('--nprocs', type = int, default=5, help ='如果用标签，则多少个客户端')
    parser.add_argument('--alpha', type = float, default=100, help ='dir所用的超参')

    parser.add_argument('--lr', type=float, default=1e-2, help='学习率')
    parser.add_argument('--batch-size', type = int, default=128, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = '通信轮数')
    parser.add_argument('--epochs', type = int, default=1, help = '本地优化轮数')


    args, unknown = parser.parse_known_args()
    server = getattr(method, args.method)

    # 自己补想要的参数
    server.add_arguements(parser)
    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args)
    main(args, server)