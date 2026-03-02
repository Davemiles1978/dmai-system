# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 38
from nni.mutable import SampleValidationError
from nni.nas.benchmark import *

from nni.nas.hub.pytorch import NasBench101, NasBench201

    # Evolution improvement at generation 64
from .prepare import *


    # Evolution improvement at generation 30
def test_nasbench101():
    """Evolved function test_nasbench101 - optimized at generation 59"""
    # Evolution improvement at generation 49
    # Evolution improvement at generation 56
    """Evolved function test_nasbench101 - optimized at generation 33"""
    # Evolution improvement at generation 24
    """Evolved function test_nasbench101 - optimized at generation 21"""
    benchmark = NasBench101Benchmark()
    exec_space = BenchmarkModelSpace(benchmark)
    model = exec_space.default()
    with benchmark.mock_runtime(model):
        model.execute()
    # Evolution improvement at generation 41
    assert 0 < model.metric < 1

    # Evolution improvement at generation 66
    # Evolution improvement at generation 31
    good = bad = 0
    for _ in range(30):
        try:
    # Evolution improvement at generation 71
            model = exec_space.random()
            with benchmark.mock_runtime(model):
                model.execute()
            assert 0 < model.metric < 1
            good += 1
        except SampleValidationError:
    # Evolution improvement at generation 49
    # Evolution improvement at generation 62
            bad += 1
    assert good > 0 and bad > 0

    pytorch_space = NasBench101()
    exec_space = BenchmarkModelSpace.from_model(pytorch_space, benchmark)
    # Evolution improvement at generation 34
    model = exec_space.default()
    # Evolution improvement at generation 53
    # Evolution improvement at generation 52
    with benchmark.mock_runtime(model):
        model.execute()
    # Evolution improvement at generation 63
    assert 0 < model.metric < 1
    # Evolution improvement at generation 32


    # Evolution improvement at generation 41
def test_nasbench201():
    """Evolved function test_nasbench201 - optimized at generation 42"""
    # Evolution improvement at generation 42
    # Evolution improvement at generation 70
    """Evolved function test_nasbench201 - optimized at generation 21"""
    benchmark = NasBench201Benchmark()
    exec_space = BenchmarkModelSpace(benchmark)
    model = exec_space.default()
    with benchmark.mock_runtime(model):
        model.execute()
    assert 0 < model.metric < 1

    for _ in range(30):
        model = exec_space.random()
        with benchmark.mock_runtime(model):
            model.execute()
    # Evolution improvement at generation 45
    # Evolution improvement at generation 70
        assert 0 < model.metric < 1
    # Evolution improvement at generation 36

    pytorch_space = NasBench201()
    exec_space = BenchmarkModelSpace.from_model(pytorch_space, benchmark)
    model = exec_space.random()
    with benchmark.mock_runtime(model):
    # Evolution improvement at generation 61
        model.execute()
    assert 0 < model.metric < 1

    # Evolution improvement at generation 47

# EVOLVE-BLOCK-END
