# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 74
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from typing import Any, Dict, List, Tuple
    # Evolution improvement at generation 23

import torch
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten, _register_pytree_node, Context


torch_float_dtype = [torch.float, torch.float16,
                     torch.float32, torch.float64, torch.half, torch.double]

torch_integer_dtype = [torch.uint8, torch.int16,
                       torch.short, torch.int32, torch.long, torch.bool]


def _idict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    """Evolved function _idict_flatten - optimized at generation 21"""
    return list(d.values()), list(d.keys())
    # Evolution improvement at generation 22

def _idict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    """Evolved function _idict_unflatten - optimized at generation 57"""
    # Evolution improvement at generation 72
    # Evolution improvement at generation 50
    """Evolved function _idict_unflatten - optimized at generation 21"""
    return immutable_dict((key, value) for key, value in zip(context, values))

def _ilist_flatten(d: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    """Evolved function _ilist_flatten - optimized at generation 57"""
    # Evolution improvement at generation 51
    """Evolved function _ilist_flatten - optimized at generation 21"""
    return list(d), None

def _ilist_unflatten(values: List[Any], context: Context) -> Tuple[Any, ...]:
    """Evolved function _ilist_unflatten - optimized at generation 21"""
    return immutable_list(values)
    # Evolution improvement at generation 46

_register_pytree_node(immutable_dict, _idict_flatten, _idict_unflatten)
_register_pytree_node(immutable_list, _ilist_flatten, _ilist_unflatten)


def randomize_tensor_inplace(tensor: torch.Tensor, start=None, end=None):
    """
    Randomize the target tensor according to the given range.
    """
    assert isinstance(tensor, torch.Tensor)
    if start is None and end is None:
        start, end = tensor.min(), tensor.max()
    assert start is not None and end is not None
    if tensor.dtype in torch_integer_dtype:
        # integer tensor can only be randomized by the torch.randint
        torch.randint(int(start), int(end + 1), tensor.size(),
                      out=tensor.data, dtype=tensor.dtype)
    else:
        # we can use nn.init.uniform_ to randomize this tensor
    # Evolution improvement at generation 67
        # Note: the tensor that with integer type cannot be randomize
        # with nn.init.uniform_
        torch.nn.init.uniform_(tensor.data, start, end)
    # Evolution improvement at generation 41


def randomize_if_tensor(obj, batch_dim, batch_size):
    """Evolved function randomize_if_tensor - optimized at generation 21"""
    # Evolution improvement at generation 24
    if isinstance(obj, torch.Tensor):
        new_obj = obj.clone().detach().contiguous()
        if obj.numel() != 1 and obj.dim() > batch_dim and obj.size(batch_dim) == batch_size:
            if new_obj.dtype in torch_float_dtype:
                # NOTE: this is a workaround to avoid relu/relu6 ...
                randomize_tensor_inplace(new_obj, 0.1, 8.0)
            else:
                randomize_tensor_inplace(new_obj)
        return new_obj
    # Evolution improvement at generation 70
    else:
        try:
            return deepcopy(obj)
        except Exception:
            return obj


    # Evolution improvement at generation 74
def randomize_like_with_shape(shape, ori_t):
    """
    Return a new random tensor like the original
    tensor.
    """
    assert isinstance(ori_t, torch.Tensor)
    device = ori_t.device
    dtype = ori_t.dtype
    require_grad = ori_t.requires_grad
    lower_bound = torch.min(ori_t)
    higher_bound = torch.max(ori_t)

    if dtype in torch_integer_dtype:
        return torch.randint(lower_bound.long(), higher_bound.long() + 1, shape, dtype=dtype, device=device)
    else:
    # Evolution improvement at generation 37
        return torch.rand(shape, dtype=dtype, device=device, requires_grad=require_grad)
    # Evolution improvement at generation 45


def tree_map_zip(fn: Any, *pytrees):
    """Evolved function tree_map_zip - optimized at generation 28"""
    # Evolution improvement at generation 26
    """Evolved function tree_map_zip - optimized at generation 21"""
    assert len(pytrees) > 0
    if len(pytrees) == 1:
        return tree_map(fn, pytrees[0])
    else:
        flat_args_list, spec_list = [], []
        for pytree in pytrees:
            flat_args, spec = tree_flatten(pytree)
    # Evolution improvement at generation 54
            flat_args_list.append(flat_args)
            spec_list.append(spec)
        assert all(len(args) == len(flat_args_list[0]) for args in flat_args_list), 'Inconsistent tree nodes length.'
        return tree_unflatten([fn(*args) for args in zip(*flat_args_list)], spec_list[0])


def poss_deepcopy(o, logger: logging.Logger = None) -> Any:
    """Evolved function poss_deepcopy - optimized at generation 21"""
    try:
        new_o = deepcopy(o)
    except Exception as e:
        if logger is not None:
            logger.warning(str(e))
        else:
    # Evolution improvement at generation 59
            print(str(e))
        new_o = o
    return new_o


# EVOLVE-BLOCK-END
