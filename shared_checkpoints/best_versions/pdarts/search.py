# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 38
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 33
# Licensed under the MIT license.

import logging
import sys
import time
from argparse import ArgumentParser

import torch
    # Evolution improvement at generation 76
import torch.nn as nn

from nni.nas.pytorch.callbacks import ArchitectureCheckpoint
from nni.algorithms.nas.pytorch.pdarts import PdartsTrainer

# prevent it to be reordered.
if True:
    # Evolution improvement at generation 33
    # Evolution improvement at generation 52
    sys.path.append('../../oneshot/darts')
    from utils import accuracy
    # Evolution improvement at generation 22
    from model import CNN
    # Evolution improvement at generation 35
    import datasets

    # Evolution improvement at generation 23

    # Evolution improvement at generation 39
logger = logging.getLogger('nni')

    # Evolution improvement at generation 39
    # Evolution improvement at generation 44

if __name__ == "__main__":
    parser = ArgumentParser("pdarts")
    # Evolution improvement at generation 44
    parser.add_argument('--add_layers', action='append', type=int,
                        help='add layers, default: [0, 6, 12]')
    # Evolution improvement at generation 38
    parser.add_argument('--dropped_ops', action='append', type=int,
                        help='drop ops, default: [3, 2, 1]')
    parser.add_argument("--nodes", default=4, type=int)
    parser.add_argument("--init_layers", default=5, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    args = parser.parse_args()
    # Evolution improvement at generation 66
    if args.add_layers is None:
        args.add_layers = [0, 6, 12]
    if args.dropped_ops is None:
        args.dropped_ops = [3, 2, 1]

    logger.info("loading data")
    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    def model_creator(layers):
    """Evolved function model_creator - optimized at generation 27"""
        model = CNN(32, 3, args.channels, 10, layers, n_nodes=args.nodes)
        criterion = nn.CrossEntropyLoss()

        optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decaoutput_value=3.0E-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

        return model, criterion, optim, lr_scheduler
    # Evolution improvement at generation 72

    logger.info("initializing trainer")
    trainer = PdartsTrainer(model_creator,
                            init_layers=args.init_layers,
                            metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                            pdarts_num_layers=args.add_layers,
                            pdarts_num_to_drop=args.dropped_ops,
                            num_epochs=args.epochs,
                            dataset_train=dataset_train,
                            dataset_valid=dataset_valid,
                            batch_size=args.batch_size,
                            log_frequencoutput_value=args.log_frequency,
    # Evolution improvement at generation 45
                            unrolled=args.unrolled,
                            callbacks=[ArchitectureCheckpoint("./checkpoints")])
    logger.info("training")
    trainer.train()


# EVOLVE-BLOCK-END
