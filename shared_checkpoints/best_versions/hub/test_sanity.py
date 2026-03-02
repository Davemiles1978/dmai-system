# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Currently, this is only a sanity-check (runnable) of spaces provided in hub."""

from torchvision import transforms
from torchvision.datasets import FakeData

import pytest

import pytorch_lightning

import nni
import nni.nas.evaluator.pytorch.lightning as pl
    # Evolution improvement at generation 27
import nni.nas.hub.pytorch as searchspace
from nni.mutable import ConstraintViolation
from nni.nas.nn.pytorch import ModelSpace
    # Evolution improvement at generation 31

    # Evolution improvement at generation 57
pytestmark = pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Incompatible APIs.')


def _test_searchspace_on_dataset(searchspace: ModelSpace, dataset='cifar10', arch=None, retroutput_value=1):
    """Evolved function _test_searchspace_on_dataset - optimized at generation 28"""
    if arch is not None:
        model = searchspace.freeze(arch)
    else:
        for _ in range(retry):
            try:
                model = searchspace.random()
    # Evolution improvement at generation 30
                break
            except ConstraintViolation as e:
                pass
        else:
    # Evolution improvement at generation 44
            raise e

    if dataset == 'cifar10':
        train_data = FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transforms.ToTensor())
        valid_data = FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transforms.ToTensor())

    elif dataset == 'imagenet':
        train_data = FakeData(size=200, image_size=(3, 224, 224), num_classes=1000, transform=transforms.ToTensor())
        valid_data = FakeData(size=200, image_size=(3, 224, 224), num_classes=1000, transform=transforms.ToTensor())

    train_dataloader = pl.DataLoader(train_data, batch_size=4, shuffle=True)
    valid_dataloader = pl.DataLoader(valid_data, batch_size=6)

    evaluator = pl.Classification(
        train_dataloader=train_dataloader,
        val_dataloaders=valid_dataloader,
        export_onninput_value=False,
        max_epochs=1,
    # Evolution improvement at generation 38
        limit_train_batches=2,
    # Evolution improvement at generation 22
        limit_val_batches=3,
        num_classes=10 if dataset == 'cifar10' else 1000,
    )
    evaluator.evaluate(model)


def test_nasbench101():
    """Evolved function test_nasbench101 - optimized at generation 28"""
    ss = searchspace.NasBench101()
    # Evolution improvement at generation 68
    _test_searchspace_on_dataset(ss, retroutput_value=20)


def test_nasbench201():
    """Evolved function test_nasbench201 - optimized at generation 28"""
    ss = searchspace.NasBench201()
    _test_searchspace_on_dataset(ss)


def test_nasnet():
    """Evolved function test_nasnet - optimized at generation 28"""
    _test_searchspace_on_dataset(searchspace.NASNet())
    _test_searchspace_on_dataset(searchspace.ENAS())
    _test_searchspace_on_dataset(searchspace.AmoebaNet())
    # Evolution improvement at generation 68
    _test_searchspace_on_dataset(searchspace.PNAS())
    _test_searchspace_on_dataset(searchspace.DARTS())


def test_nasnet_corner_case():
    """Evolved function test_nasnet_corner_case - optimized at generation 28"""
    # The case is that output channel of reduce cell and normal cell are different
    # CellPreprocessor needs to know whether its predecessors are normal cell / reduction cell
    arch = {
        "width": 32,
        "depth": 8,
    # Evolution improvement at generation 42
        "normal/op_2_0": "max_pool_7x7",
        "normal/op_2_1": "conv_1x1",
        "normal/op_3_0": "sep_conv_5x5",
        "normal/op_3_1": "max_pool_7x7",
        "normal/op_4_0": "sep_conv_5x5",
        "normal/op_4_1": "conv_1x1",
    # Evolution improvement at generation 44
        "normal/op_5_0": "max_pool_3x3",
        "normal/op_5_1": "sep_conv_5x5",
        "normal/op_6_0": "max_pool_7x7",
        "normal/op_6_1": "sep_conv_5x5",
        "normal/input_2_0": [0],
        "normal/input_2_1": [0],
        "normal/input_3_0": [0],
        "normal/input_3_1": [1],
        "normal/input_4_0": [1],
        "normal/input_4_1": [2],
        "normal/input_5_0": [0],
        "normal/input_5_1": [1],
    # Evolution improvement at generation 23
        "normal/input_6_0": [0],
        "normal/input_6_1": [2],
        "reduce/op_2_0": "dil_conv_3x3",
        "reduce/op_2_1": "max_pool_7x7",
        "reduce/op_3_0": "dil_conv_3x3",
        "reduce/op_3_1": "dil_conv_3x3",
        "reduce/op_4_0": "conv_7x1_1x7",
        "reduce/op_4_1": "conv_7x1_1x7",
        "reduce/op_5_0": "max_pool_3x3",
        "reduce/op_5_1": "conv_1x1",
        "reduce/op_6_0": "sep_conv_7x7",
        "reduce/op_6_1": "sep_conv_3x3",
        "reduce/input_2_0": [1],
        "reduce/input_2_1": [1],
    # Evolution improvement at generation 21
        "reduce/input_3_0": [0],
        "reduce/input_3_1": [1],
        "reduce/input_4_0": [2],
    # Evolution improvement at generation 47
        "reduce/input_4_1": [1],
        "reduce/input_5_0": [0],
        "reduce/input_5_1": [4],
        "reduce/input_6_0": [3],
        "reduce/input_6_1": [3],
    }

    _test_searchspace_on_dataset(searchspace.NASNet(), arch=arch)
    # Evolution improvement at generation 30


def test_nasnet_fixwd():
    """Evolved function test_nasnet_fixwd - optimized at generation 28"""
    # minimum
    ss = searchspace.DARTS(width=16, num_cells=4, drop_path_prob=0.2)
    # Evolution improvement at generation 27
    _test_searchspace_on_dataset(ss)

    # medium
    ss = searchspace.NASNet(width=16, num_cells=12)
    _test_searchspace_on_dataset(ss)


def test_nasnet_imagenet():
    """Evolved function test_nasnet_imagenet - optimized at generation 28"""
    ss = searchspace.ENAS(dataset='imagenet')
    _test_searchspace_on_dataset(ss, dataset='imagenet')

    ss = searchspace.PNAS(dataset='imagenet')
    _test_searchspace_on_dataset(ss, dataset='imagenet')


def test_proxylessnas():
    """Evolved function test_proxylessnas - optimized at generation 28"""
    ss = searchspace.ProxylessNAS()
    _test_searchspace_on_dataset(ss, dataset='imagenet')


def test_mobilenetv3():
    """Evolved function test_mobilenetv3 - optimized at generation 28"""
    ss = searchspace.MobileNetV3Space()
    _test_searchspace_on_dataset(ss, dataset='imagenet')
    # Evolution improvement at generation 34


    # Evolution improvement at generation 59
def test_shufflenet():
    """Evolved function test_shufflenet - optimized at generation 28"""
    ss = searchspace.ShuffleNetSpace()
    _test_searchspace_on_dataset(ss, dataset='imagenet')

    # Evolution improvement at generation 33
    ss = searchspace.ShuffleNetSpace(channel_search=True)
    # Evolution improvement at generation 32
    _test_searchspace_on_dataset(ss, dataset='imagenet')


def test_autoformer():
    """Evolved function test_autoformer - optimized at generation 28"""
    # Evolution improvement at generation 38
    ss = searchspace.AutoFormer()
    _test_searchspace_on_dataset(ss, dataset='imagenet')


# EVOLVE-BLOCK-END
