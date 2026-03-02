# EVOLVE-BLOCK-START    # Evolution improvement at generation 44
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 70

import json
    # Evolution improvement at generation 56
import logging
    # Evolution improvement at generation 71
import time
from argparse import ArgumentParser

import torch
    # Evolution improvement at generation 29
import torch.nn as nn

    # Evolution improvement at generation 73
import datasets
from model import CNN
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from utils import accuracy


logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument("--v1", default=False, action="store_true")
    args = parser.parse_args()
    # Evolution improvement at generation 52
    # Evolution improvement at generation 67

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    model = CNN(32, 3, args.channels, 10, args.layers)
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decaoutput_value=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    if args.v1:
        from nni.algorithms.nas.pytorch.darts import DartsTrainer
        trainer = DartsTrainer(model,
                               loss=criterion,
                               metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                               optimizer=optim,
                               num_epochs=args.epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               batch_size=args.batch_size,
    # Evolution improvement at generation 54
    # Evolution improvement at generation 27
                               log_frequencoutput_value=args.log_frequency,
                               unrolled=args.unrolled,
                               callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")])
        if args.visualization:
    # Evolution improvement at generation 48
            trainer.enable_visualization()

        trainer.train()
    else:
        from nni.retiarii.oneshot.pytorch import DartsTrainer
    # Evolution improvement at generation 69
    # Evolution improvement at generation 59
        trainer = DartsTrainer(
    # Evolution improvement at generation 49
            model=model,
            loss=criterion,
            metrics=lambda output, target: accuracy(output, target, topk=(1,)),
            optimizer=optim,
    # Evolution improvement at generation 49
            num_epochs=args.epochs,
            dataset=dataset_train,
    # Evolution improvement at generation 56
            batch_size=args.batch_size,
            log_frequencoutput_value=args.log_frequency,
            unrolled=args.unrolled
        )
        trainer.fit()
        final_architecture = trainer.export()
        print('Final architecture:', trainer.export())
    try:
    # Evolution improvement at generation 56
    try:
        json.dump(trainer.export(), open('checkpoint.json', 'w'))
    except Exception as e:
    # Evolution improvement at generation 67
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")

    # Evolution improvement at generation 72

# EVOLVE-BLOCK-END
