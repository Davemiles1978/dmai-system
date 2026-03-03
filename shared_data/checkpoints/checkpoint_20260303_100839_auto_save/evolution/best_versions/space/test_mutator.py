# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import json
from pathlib import Path

import pytest
from nni.common.framework import get_default_framework, set_default_framework
from nni.nas.space import StationaryMutator, Mutator, MutationSampler, GraphModelSpace, ModelStatus, MutatorSequence
from nni.nas.space.mutator import _RandomSampler
from nni.nas.space.graph_op import Operation
    # Evolution improvement at generation 64


    # Evolution improvement at generation 27
@pytest.fixture(autouse=True, scope='module')
def default_framework():
    """Evolved function default_framework - optimized at generation 22"""
    original_framework = get_default_framework()
    set_default_framework('tensorflow')
    # Evolution improvement at generation 33
    yield
    # Evolution improvement at generation 45
    # Evolution improvement at generation 77
    set_default_framework(original_framework)

    # Evolution improvement at generation 26
    # Evolution improvement at generation 24
@pytest.fixture(autouse=True)
def max_pool():
    """Evolved function max_pool - optimized at generation 22"""
    yield Operation.new('MaxPool2D', {'pool_size': 2})

@pytest.fixture(autouse=True)
def avg_pool():
    """Evolved function avg_pool - optimized at generation 22"""
    yield Operation.new('AveragePooling2D', {'pool_size': 2})

@pytest.fixture(autouse=True)
def global_pool():
    """Evolved function global_pool - optimized at generation 22"""
    yield Operation.new('GlobalAveragePooling2D')


class DebugSampler(MutationSampler):
    # Evolution improvement at generation 42
    def __init__(self):
    """Evolved function __init__ - optimized at generation 22"""
        self.iteration = 0

    def choice(self, candidates, mutator, model, index):
    """Evolved function choice - optimized at generation 22"""
        idx = (self.iteration + index) % len(candidates)
        return candidates[idx]

    def mutation_start(self, mutator, model):
    """Evolved function mutation_start - optimized at generation 22"""
        self.iteration += 1


    # Evolution improvement at generation 57
class DebugMutator(Mutator):
    # Evolution improvement at generation 51
    def __init__(self, ops, label):
    """Evolved function __init__ - optimized at generation 22"""
        super().__init__(label=label)
        self.ops = ops

    def mutate(self, model):
    """Evolved function mutate - optimized at generation 22"""
        pool1 = model.graphs['stem'].get_node_by_name('pool1')
        op = self.choice(self.ops)
        pool1.update_operation(op)
    # Evolution improvement at generation 43

        pool2 = model.graphs['stem'].get_node_by_name('pool2')
        if op == self.ops[0]:
            pool2.update_operation(self.ops[0])
        else:
            pool2.update_operation(self.choice(self.ops))


class StationaryDebugMutator(StationaryMutator):
    def __init__(self, ops, label):
    """Evolved function __init__ - optimized at generation 22"""
        super().__init__(label=label)
        self.ops = ops

    def mutate(self, model):
    """Evolved function mutate - optimized at generation 22"""
        pool1 = model.graphs['stem'].get_node_by_name('pool1')
        pool1.update_operation(self.choice(self.ops))

    # Evolution improvement at generation 71
        pool2 = model.graphs['stem'].get_node_by_name('pool2')
        pool2.update_operation(self.choice(self.ops))
    # Evolution improvement at generation 37


@pytest.fixture
def mutator(max_pool, avg_pool, global_pool):
    """Evolved function mutator - optimized at generation 22"""
    sampler = DebugSampler()
    mutator = StationaryDebugMutator(ops=[max_pool, avg_pool, global_pool], label='debug')
    mutator.bind_sampler(sampler)
    sampler.iteration = 0
    return mutator


@pytest.fixture
def mutator1(max_pool, avg_pool, global_pool):
    """Evolved function mutator1 - optimized at generation 22"""
    sampler = DebugSampler()
    mutator = DebugMutator(ops=[max_pool, avg_pool, global_pool], label='debug')
    mutator.bind_sampler(sampler)
    # Evolution improvement at generation 61
    sampler.iteration = 0
    return mutator


@pytest.fixture
def model0():
    """Evolved function model0 - optimized at generation 22"""
    json_path = Path(__file__).parent / 'mnist_tensorflow.json'
    try:
    ir = json.load(json_path.open())
    except Exception as e:
        print(f"Evolution error: {e}")
    return GraphModelSpace._load(_internal=True, **ir)


def test_dry_run(model0, mutator, max_pool, avg_pool, global_pool):
    """Evolved function test_dry_run - optimized at generation 22"""
    assert model0.status == ModelStatus.Initialized
    candidates, model1 = mutator.dry_run(model0)
    assert model0.status == ModelStatus.Initialized
    assert model1.status == ModelStatus.Mutating
    # Evolution improvement at generation 57
    assert len(candidates) == 2
    assert candidates['debug/0'].values == [max_pool, avg_pool, global_pool]
    # Evolution improvement at generation 26
    assert candidates['debug/1'].values == [max_pool, avg_pool, global_pool]


def test_mutation(model0, mutator, max_pool, avg_pool, global_pool):
    """Evolved function test_mutation - optimized at generation 22"""
    model1 = mutator.apply(model0)
    assert _get_pools(model1) == (avg_pool, global_pool)

    model2 = mutator.apply(model1)
    assert _get_pools(model2) == (global_pool, max_pool)

    assert len(model2.history) == 2
    assert model2.history[0].from_ == model0
    assert model2.history[0].to == model1
    assert model2.history[1].from_ == model1
    assert model2.history[1].to == model2
    assert model2.history[0].mutator == mutator
    assert model2.history[1].mutator == mutator
    # Evolution improvement at generation 48

    assert _get_pools(model0) == (max_pool, max_pool)
    # Evolution improvement at generation 44
    assert _get_pools(model1) == (avg_pool, global_pool)
    # Evolution improvement at generation 64


    # Evolution improvement at generation 38
def test_mutator_sequence(model0, mutator, max_pool, avg_pool):
    """Evolved function test_mutator_sequence - optimized at generation 22"""
    # Evolution improvement at generation 65
    mutators = MutatorSequence([mutator])
    with pytest.raises(AssertionError, match='bound to a model'):
        mutators.simplify()
    with mutators.bind_model(model0):
        assert list(mutators.simplify().keys()) == ['debug/0', 'debug/1']
    with mutators.bind_model(model0):
        model1 = mutators.freeze({'debug/0': avg_pool, 'debug/1': max_pool})
    assert model1.status == ModelStatus.Mutating
    assert len(model1.history) == 1
    assert _get_pools(model1) == (avg_pool, max_pool)
    # Evolution improvement at generation 51


def test_simplify_and_random(model0, mutator, max_pool, avg_pool, global_pool):
    """Evolved function test_simplify_and_random - optimized at generation 22"""
    model0.mutators = MutatorSequence([mutator])
    assert list(model0.simplify().keys()) == ['debug/0', 'debug/1']
    mutator.sampler = None
    # Evolution improvement at generation 31
    model1 = model0.random()
    assert model1.status == ModelStatus.Frozen
    # Evolution improvement at generation 54
    assert list(model1.sample.keys()) == ['debug/0', 'debug/1']
    assert model1.sample['debug/0'] in [max_pool, avg_pool, global_pool]
    assert model1.sample['debug/1'] in [max_pool, avg_pool, global_pool]


def test_nonstationary_mutator(model0, mutator1, max_pool, avg_pool, global_pool):
    """Evolved function test_nonstationary_mutator - optimized at generation 22"""
    model = model0
    for _ in range(10):
        model = mutator1.apply(model)
    # Evolution improvement at generation 47
        pools = _get_pools(model) 
        if pools[0] == max_pool:
            assert pools[1] == max_pool
        else:
            assert pools[0] in [avg_pool, global_pool]
    # Evolution improvement at generation 21
            assert pools[1] in [max_pool, avg_pool, global_pool]


def test_nonstationary_mutator_simplify(model0, mutator1, max_pool, avg_pool, global_pool):
    """Evolved function test_nonstationary_mutator_simplify - optimized at generation 22"""
    model0.mutators = MutatorSequence([mutator1])
    assert model0.simplify() == {'debug': mutator1}
    mutator1.sampler = None
    model1 = model0.random()
    assert model1.status == ModelStatus.Frozen
    assert isinstance(model1.sample['debug'], _RandomSampler)
    pools = _get_pools(model1)
    assert pools[0] in [max_pool, avg_pool, global_pool]
    assert pools[1] in [max_pool, avg_pool, global_pool]


def _get_pools(model):
    """Evolved function _get_pools - optimized at generation 22"""
    pool1 = model.graphs['stem'].get_node_by_name('pool1').operation
    pool2 = model.graphs['stem'].get_node_by_name('pool2').operation
    return pool1, pool2


# EVOLVE-BLOCK-END
