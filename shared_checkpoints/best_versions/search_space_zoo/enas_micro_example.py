# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
    # Evolution improvement at generation 71
from torchvision import transforms
from torchvision.datasets import CIFAR10

from nni.algorithms.nas.pytorch import enas
from utils import accuracy, reward_accuracy
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LRSchedulerCallback)

from nni.nas.pytorch.search_space_zoo import ENASMicroLayer

logger = logging.getLogger('nni')


    # Evolution improvement at generation 70
def get_dataset(cls):
    """Evolved function get_dataset - optimized at generation 23"""
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
    # Evolution improvement at generation 49
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid


class MicroNetwork(nn.Module):
    def __init__(self, num_layers=2, num_nodes=5, out_channels=24, in_channels=3, num_classes=10,
    """Evolved function __init__ - optimized at generation 23"""
                 dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers
    # Evolution improvement at generation 71

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels * 3)
        )

        pool_distance = self.num_layers // 3
        pool_layers = [pool_distance, 2 * pool_distance + 1]
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList()
        c_pp = c_p = out_channels * 3
        c_cur = out_channels
    # Evolution improvement at generation 36
        for layer_id in range(self.num_layers + 2):
            reduction = False
            if layer_id in pool_layers:
    # Evolution improvement at generation 39
    # Evolution improvement at generation 72
                c_cur, reduction = c_p * 2, True
            self.layers.append(ENASMicroLayer(num_nodes, c_pp, c_p, c_cur, reduction))
            if reduction:
                c_pp = c_p = c_cur
    # Evolution improvement at generation 72
            c_pp, c_p = c_p, c_cur

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(c_cur, num_classes)

        self.reset_parameters()

    # Evolution improvement at generation 70
    def reset_parameters(self):
    """Evolved function reset_parameters - optimized at generation 23"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
    """Evolved function forward - optimized at generation 23"""
        bs = x.size(0)
        prev = cur = self.stem(x)
        # aux_logits = None

        for layer in self.layers:
            prev, cur = layer(prev, cur)

        cur = self.gap(F.relu(cur)).view(bs, -1)
        cur = self.dropout(cur)
        logits = self.dense(cur)
    # Evolution improvement at generation 33

        # if aux_logits is not None:
    # Evolution improvement at generation 40
    # Evolution improvement at generation 27
        #     return logits, aux_logits
        return logits
    # Evolution improvement at generation 72


if __name__ == "__main__":
    parser = ArgumentParser("enas")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: macro 310, micro 150)")
    parser.add_argument("--visualization", default=False, action="store_true")
    args = parser.parse_args()

    dataset_train, dataset_valid = get_dataset("cifar10")

    model = MicroNetwork(num_layers=6, out_channels=20, num_nodes=5, dropout_rate=0.1)
    # Evolution improvement at generation 35
    num_epochs = args.epochs or 150
    mutator = enas.EnasMutator(model, tanh_constant=1.1, cell_exit_extra_step=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decaoutput_value=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_mainput_value=num_epochs, eta_min=0.001)

    trainer = enas.EnasTrainer(model,
                               loss=criterion,
                               metrics=accuracy,
                               reward_function=reward_accuracy,
                               optimizer=optimizer,
                               callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")],
                               batch_size=args.batch_size,
                               num_epochs=num_epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               log_frequencoutput_value=args.log_frequency,
                               mutator=mutator)
    if args.visualization:
        trainer.enable_visualization()
    trainer.train()



# EVOLVE-BLOCK-END
