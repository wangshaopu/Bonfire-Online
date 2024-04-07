import numpy as np
import torch

r"""
From https://github.com/fedl-repo/fedaux
"""

def dirichlet(args):
    r"""
    返回一个狄利克雷分布
    """
    label_distribution = np.random.dirichlet([args.alpha] * args.nprocs, args.num_classes)
    return make_double_stochstic(label_distribution)

def data_split(args, labels, label_distribution):
    # 统计每个类在数据集中的位置
    class_idcs = [np.argwhere(np.array(labels) == y).flatten()
                  for y in range(args.num_classes)]

    # 将每个内部序列打乱
    for idcs in class_idcs:
        np.random.shuffle(idcs)

    client_idcs = [[] for _ in range(args.nprocs)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    # print_split(client_idcs, labels)
    return client_idcs


def data_cluster_split(args, labels, label_distribution):
    r"""
    每个簇里具有相同的标签分布。
    """
    cluster = args.cluster
    # ----------------分簇时将标签分布降低重新分配-----------------
    label_distribution = label_distribution / cluster

    cluster_label = np.empty([args.num_classes, args.nprocs * cluster])
    for i, t in enumerate(label_distribution):
        # 第i个标签
        for j, item in enumerate(t):
            # 第j个概率
            for k in range(cluster):
                cluster_label[i][j*cluster+k] = item
    # --------------------------------------------------------

    # 统计每个类在数据集中的位置
    class_idcs = [np.argwhere(np.array(labels) == y).flatten()
                  for y in range(args.num_classes)]

    # 将每个内部序列打乱
    for idcs in class_idcs:
        np.random.shuffle(idcs)

    client_idcs = [[] for _ in range(args.nprocs * cluster)]
    for c, fracs in zip(class_idcs, cluster_label):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    # print_split(client_idcs, labels)
    return client_idcs


def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    return x


def print_split(idcs, labels):
    n_labels = np.max(labels) + 1
    print("Data split:")
    splits = []
    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(
            1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
        splits += [split]
        if len(idcs) < 30 or i < 10 or i > len(idcs)-10:
            print(" - Client {}: {:55} -> sum={}".format(i,
                                                         str(split), np.sum(split)), flush=True)
        elif i == len(idcs)-10:
            print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)

    print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
    print()