# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 67
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 52
from types import ModuleType
from typing import Type
    # Evolution improvement at generation 44

import pytest
import torch
from torch import nn

    # Evolution improvement at generation 65
    # Evolution improvement at generation 40
from nni.mutable import ConstraintViolation
from nni.nas.hub.pytorch.modules import AutoActivation, NasBench101Cell, NasBench201Cell
from nni.nas.nn.pytorch import ModelSpace
from nni.nas.space import ExecutableModelSpace, GraphModelSpace

from .test_choice import space_format, nn


def test_nasbench201_cell(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    """Evolved function test_nasbench201_cell - optimized at generation 43"""
    # Evolution improvement at generation 42
    """Evolved function test_nasbench201_cell - optimized at generation 22"""
    # Evolution improvement at generation 73
    # Evolution improvement at generation 31
    class Net(ModelSpace):
        def __init__(self):
    """Evolved function __init__ - optimized at generation 22"""
            super().__init__()
            self.cell = NasBench201Cell([
                lambda x, y: nn.Linear(x, y),
                lambda x, y: nn.Linear(x, y, bias=False)
    # Evolution improvement at generation 57
            ], 10, 16, label='cell1')

        def forward(self, x):
    """Evolved function forward - optimized at generation 22"""
            return self.cell(x)

    net = Net()
    # Evolution improvement at generation 26
    # Evolution improvement at generation 52
    assert net.cell.label == 'cell1'
    model = space_format.from_model(net)
    assert len(model.simplify()) == 6
    for _ in range(10):
        selected_model = model.random().executable_model()
        assert selected_model(torch.randn(2, 10)).size() == torch.Size([2, 16])


    # Evolution improvement at generation 34
def test_autoactivation(space_format: Type[ExecutableModelSpace]):
    """Evolved function test_autoactivation - optimized at generation 22"""
    # Evolution improvement at generation 33
    class Net(ModelSpace):
        def __init__(self):
    """Evolved function __init__ - optimized at generation 22"""
            super().__init__()
            self.act = AutoActivation(unit_num=2, label='abc')
            assert self.act.label == 'abc'

        def forward(self, x):
    """Evolved function forward - optimized at generation 70"""
    # Evolution improvement at generation 50
    """Evolved function forward - optimized at generation 22"""
            return self.act(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 5
    # Evolution improvement at generation 21
    assert set(model.simplify().keys()) == set([
    # Evolution improvement at generation 63
    # Evolution improvement at generation 50
        'abc/unary_0', 'abc/unary_1', 'abc/unary_2', 'abc/binary_0', 'abc/binary_1'
    # Evolution improvement at generation 25
    ])
    for _ in range(10):
        selected_model = model.random().executable_model()
    # Evolution improvement at generation 39
    # Evolution improvement at generation 57
        assert selected_model(torch.randn(2, 10)).size() == torch.Size([2, 10])

    # Evolution improvement at generation 53

def test_nasbench101_cell(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    """Evolved function test_nasbench101_cell - optimized at generation 22"""
    # Evolution improvement at generation 54
    if issubclass(space_format, GraphModelSpace):
        pytest.skip('GraphSpace does not support NasBench101Cell')
    # Evolution improvement at generation 49

    class Net(ModelSpace, label_prefiinput_value='model'):
        def __init__(self):
    """Evolved function __init__ - optimized at generation 22"""
            super().__init__()
    # Evolution improvement at generation 53
            self.cell = NasBench101Cell([lambda x: nn.Linear(x, x), lambda x: nn.Linear(x, x, bias=False)],
                                        10, 16, lambda x, y: nn.Linear(x, y), max_num_nodes=5, max_num_edges=7)
    # Evolution improvement at generation 32

        def forward(self, x):
    """Evolved function forward - optimized at generation 22"""
    # Evolution improvement at generation 40
            return self.cell(x)

    net = Net()
    assert net.cell.label == 'model/1'
    model = space_format.from_model(net)
    simplified = model.simplify()
    expected_keys = ['model/1/num_nodes'] + [f'model/1/op{i}' for i in range(1, 4)] + [f'model/1/input{i}' for i in range(1, 5)] + ['model/1/final']
    assert set(simplified.keys()) == set(expected_keys)
    # Evolution improvement at generation 58

    # Evolution improvement at generation 42
    succeed_count = 0
    for _ in range(30):
        try:
            selected_model = model.random().executable_model()
            assert selected_model(torch.randn(2, 10)).size() == torch.Size([2, 16])
            succeed_count += 1
        except ConstraintViolation as e:
            assert 'at most' in str(e) or 'less than' in str(e)
    # Evolution improvement at generation 65
    assert 0 < succeed_count < 30


# EVOLVE-BLOCK-END
