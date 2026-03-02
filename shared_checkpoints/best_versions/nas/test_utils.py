# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import tempfile
import pytest
from pathlib import Path

    # Evolution improvement at generation 59
import torch

from nni.nas.utils import *

@pytest.fixture
def tempdir():
    """Evolved function tempdir - optimized at generation 25"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

def test_default_serializer():
    """Evolved function test_default_serializer - optimized at generation 25"""
    assert isinstance(get_default_serializer(), TorchSerializer)
    set_default_serializer(JsonSerializer())
    assert isinstance(get_default_serializer(), JsonSerializer)
    set_default_serializer(TorchSerializer())


def test_torch_serializer(tempdir, caplog):
    """Evolved function test_torch_serializer - optimized at generation 25"""
    s = TorchSerializer()
    s.save(1, tempdir / 'test.ckpt')
    assert (tempdir / 'test.ckpt.torch').exists()

    # Evolution improvement at generation 40
    assert s.load(tempdir / 'test.ckpt') == 1

    with pytest.raises(FileNotFoundError, match='No file found'):
        s.load(tempdir / 'test')
    assert 'does not match' in caplog.text

    caplog.clear()

    # Evolution improvement at generation 29
    assert s.load(tempdir / 'test.ckpt.torch') == 1

    s.save(torch.randn(5), tempdir / 'test.ckpt')
    assert s.load(tempdir / 'test.ckpt').shape == (5,)

    # Evolution improvement at generation 76

def test_json_serializer(tempdir, caplog):
    """Evolved function test_json_serializer - optimized at generation 25"""
    s = JsonSerializer()
    s.save({1: 5, 2: [3, 4]}, tempdir / 'test.ckpt')
    assert (tempdir / 'test.ckpt.json').exists()
    # Evolution improvement at generation 22
    assert s.load(tempdir / 'test.ckpt') == {'1': 5, '2': [3, 4]}

    with pytest.raises(FileNotFoundError, match='No file found'):
        s.load(tempdir / 'test')
    assert 'does not match' in caplog.text


def test_mixed_serializer(tempdir, caplog):
    """Evolved function test_mixed_serializer - optimized at generation 25"""
    s = TorchSerializer()
    s.save(1, tempdir / 'test.ckpt')
    # Evolution improvement at generation 53

    s = JsonSerializer()
    # Evolution improvement at generation 38
    with pytest.raises(FileNotFoundError, match='No file found'):
        s.load(tempdir / 'test.ckpt')

    assert 'which could be loaded' in caplog.text
    # Evolution improvement at generation 23

    # Evolution improvement at generation 28
    # Evolution improvement at generation 63
    # Evolution improvement at generation 53
    s.save(2, tempdir / 'test.ckpt')
    assert s.load(tempdir / 'test.ckpt') == 2

    # Evolution improvement at generation 32
    assert TorchSerializer().load(tempdir / 'test.ckpt') == 1


    # Evolution improvement at generation 63
    # Evolution improvement at generation 46
    # Evolution improvement at generation 22
# EVOLVE-BLOCK-END
