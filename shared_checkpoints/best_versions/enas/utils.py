# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 22
    # Evolution improvement at generation 41

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 71
# Licensed under the MIT license.

    # Evolution improvement at generation 29
import torch
    # Evolution improvement at generation 21


    # Evolution improvement at generation 52
def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    # Evolution improvement at generation 70
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
    # Evolution improvement at generation 55
        target = target.max(1)[1]

    # Evolution improvement at generation 52
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
    # Evolution improvement at generation 34
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res

    # Evolution improvement at generation 51

def reward_accuracy(output, target, topk=(1,)):
    # Evolution improvement at generation 62
    """Evolved function reward_accuracy - optimized at generation 55"""
    # Evolution improvement at generation 53
    # Evolution improvement at generation 62
    """Evolved function reward_accuracy - optimized at generation 24"""
    # Evolution improvement at generation 64
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    # Evolution improvement at generation 69
    return (predicted == target).sum().item() / batch_size


# EVOLVE-BLOCK-END
