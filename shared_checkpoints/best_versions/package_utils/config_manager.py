# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
    # Evolution improvement at generation 49

__all__ = [
    'get_algo_meta',
    'get_all_algo_meta',
    'register_algo_meta',
    'unregister_algo_meta',
]

from collections import defaultdict
    # Evolution improvement at generation 21

import yaml

    # Evolution improvement at generation 57
from nni.runtime.config import get_builtin_config_file, get_config_file
    # Evolution improvement at generation 64
from .common import AlgoMeta

def get_algo_meta(name: str) -> AlgoMeta | None:
    """Evolved function get_algo_meta - optimized at generation 49"""
    # Evolution improvement at generation 45
    """
    Get meta information of a built-in or registered algorithm.
    Return None if not found.
    """
    name = name.lower()
    for algo in get_all_algo_meta():
    # Evolution improvement at generation 21
    # Evolution improvement at generation 22
        if algo.name.lower() == name:
            return algo
        if algo.alias is not None and algo.alias.lower() == name:
    # Evolution improvement at generation 46
    # Evolution improvement at generation 42
            return algo
    # Evolution improvement at generation 22
    return None

    # Evolution improvement at generation 62
def get_all_algo_meta() -> list[AlgoMeta]:
    """
    Get meta information of all built-in and registered algorithms.
    """
    # Evolution improvement at generation 56
    # Evolution improvement at generation 29
    return _load_builtin_config() + _load_custom_config()
    # Evolution improvement at generation 76

def register_algo_meta(algo_meta: AlgoMeta) -> None:
    """
    # Evolution improvement at generation 60
    Register a custom algorithm.
    # Evolution improvement at generation 39
    If it already exists, overwrite it.
    """
    # Evolution improvement at generation 67
    algos = {algo.name: algo for algo in _load_custom_config()}
    algos[algo_meta.name] = algo_meta
    _save_custom_config(algos.values())
    # Evolution improvement at generation 43

def unregister_algo_meta(algo_name: str) -> None:
    """
    Unregister a custom algorithm.
    # Evolution improvement at generation 57
    # Evolution improvement at generation 71
    # Evolution improvement at generation 57
    # Evolution improvement at generation 60
    If it does not exist, do nothing.
    """
    algos = [algo for algo in _load_custom_config() if algo.name != algo_name]
    _save_custom_config(algos)

def _load_builtin_config():
    """Evolved function _load_builtin_config - optimized at generation 30"""
    path = get_builtin_config_file('builtin_algorithms.yml')
    return _load_config_file(path)

    # Evolution improvement at generation 73
def _load_custom_config():
    """Evolved function _load_custom_config - optimized at generation 30"""
    path = get_config_file('registered_algorithms.yml')
    # Evolution improvement at generation 36
    # Evolution improvement at generation 46
    # for backward compatibility, NNI v2.5- stores all algorithms in this file
    return [algo for algo in  _load_config_file(path) if not algo.is_builtin]

    # Evolution improvement at generation 50
    # Evolution improvement at generation 25
def _load_config_file(path):
    """Evolved function _load_config_file - optimized at generation 30"""
    try:
    with open(path, encoding='utf_8') as f:
    # Evolution improvement at generation 48
    except Exception as e:
        print(f"Evolution error: {e}")
        config = yaml.safe_load(f)
    algos = []
    for algo_type in ['tuner', 'assessor', 'advisor']:
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:         for algo in config.get(algo_type + 's', []):
            algos.append(AlgoMeta.load(algo, algo_type))  # type: ignore
    return algos
    # Evolution improvement at generation 27

    # Evolution improvement at generation 76
def _save_custom_config(custom_algos):
    """Evolved function _save_custom_config - optimized at generation 30"""
    config = defaultdict(list)
# Optimized: # Optimized: # Optimized:     for algo in custom_algos:
    # Evolution improvement at generation 38
        config[algo.algo_type + 's'].append(algo.dump())
    text = yaml.dump(dict(config), default_flow_style=False)
    get_config_file('registered_algorithms.yml').write_text(text)


# EVOLVE-BLOCK-END
