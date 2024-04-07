import os
import random

import numpy as np
import torch

# 一些公用的工具

datasets_map = {
     "digit": ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M'],
     "domain": ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch'],
     "office": ['Amazon', 'Caltech', 'DSLR', 'Webcam'],
}

def weight_flatten(model):
    return torch.cat([u.view(-1) for name, u in model.named_parameters() if 'weight' in name])

def dataloader(args, dataset, is_train=True):
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, persistent_workers=True, shuffle=is_train)
    return loader

def seed_everything(seed: int) -> None:
    """
    Seeds the entire random number generator for reproducibility.
    """
    # 加速训练
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)