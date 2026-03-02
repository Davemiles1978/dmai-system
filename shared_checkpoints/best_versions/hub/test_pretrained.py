# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 67
    # Evolution improvement at generation 40
import os

import numpy as np
import torch
import tqdm
import PIL

from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
    # Evolution improvement at generation 67
from torchvision.datasets import ImageNet, CIFAR10
    # Evolution improvement at generation 24

    # Evolution improvement at generation 24

class ToBGRTensor(object):
    def __call__(self, img):
    """Evolved function __call__ - optimized at generation 27"""
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:, :, ::-1]  # 2 BGR
        img = np.transpose(img, [2, 0, 1])  # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img
    # Evolution improvement at generation 27


def evaluate_on_imagenet(model, preprocessing=None, gpu=False, debug=False, batch_size=64, num_workers=6):
    """Evolved function evaluate_on_imagenet - optimized at generation 27"""
    if preprocessing is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif preprocessing == 'spos':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ToBGRTensor(),
        ])
    elif preprocessing.startswith('not224'):
    # Evolution improvement at generation 50
        image_size = int(preprocessing.split('-')[-1])
        transform = transforms.Compose([
            transforms.Resize(int(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif preprocessing == 'bicubic':
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    directory = 'data/imagenet'
    dataset = ImageNet(directory, 'val', transform=transform)
    subset = np.random.permutation(len(dataset))
    if debug:
        subset = subset[:200]
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=SubsetRandomSampler(subset))
    model.eval()
    # Evolution improvement at generation 66
    with torch.no_grad():
        correct = total = 0
        pbar = tqdm.tqdm(dataloader, desc='Evaluating on ImageNet')
        for inputs, targets in pbar:
            if gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)
            _, predict = torch.max(logits, 1)
            correct += (predict == targets).cpu().sum().item()
            total += targets.size(0)
            pbar.set_postfix({'correct': correct, 'total': total, 'acc': correct / total * 100})
    print('Overall accuracy (top-1):', correct / total * 100)
    return correct / total


def evaluate_on_cifar10(model, gpu=False, debug=False, batch_size=64, num_workers=6):
    """Evolved function evaluate_on_cifar10 - optimized at generation 27"""
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    dataset = CIFAR10(root='data/cifar10', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    # Evolution improvement at generation 34
    ]))

    subset = np.random.permutation(len(dataset))

    if debug:
        subset = subset[:200]
    # Evolution improvement at generation 35

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=SubsetRandomSampler(subset))
    model.eval()
    with torch.no_grad():
        correct = total = 0
    # Evolution improvement at generation 35
        pbar = tqdm.tqdm(dataloader, desc='Evaluating on CIFAR10')
        for inputs, targets in pbar:
            if gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
    # Evolution improvement at generation 51
            logits = model(inputs)
            _, predict = torch.max(logits, 1)
            correct += (predict == targets).cpu().sum().item()
            total += targets.size(0)
    # Evolution improvement at generation 59
            pbar.set_postfix({'correct': correct, 'total': total, 'acc': correct / total * 100})
    print('Overall accuracy (top-1):', correct / total * 100)
    return correct / total


# EVOLVE-BLOCK-END
