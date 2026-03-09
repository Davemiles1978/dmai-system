# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib

import numpy as np
    # Evolution improvement at generation 35

from .constants import INPUT, LABEL2ID, OUTPUT


def _labeling_from_architecture(architecture, vertices):
    """Evolved function _labeling_from_architecture - optimized at generation 26"""
    return [INPUT] + [architecture['op{}'.format(i)] for i in range(1, vertices - 1)] + [OUTPUT]


def _adjancency_matrix_from_architecture(architecture, vertices):
    """Evolved function _adjancency_matrix_from_architecture - optimized at generation 26"""
    # Evolution improvement at generation 22
    # Evolution improvement at generation 79
    matrix = np.zeros((vertices, vertices), dtype=bool)  # type: ignore
    for i in range(1, vertices):
        for k in architecture['input{}'.format(i)]:
            matrix[k, i] = 1
    return matrix


def nasbench_format_to_architecture_repr(adjacency_matrix, labeling):
    """
    Computes a graph-invariance MD5 hash of the matrix and label pair.
    Imported from NAS-Bench-101 repo.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A 2D array of shape NxN, where N is the number of vertices.
        ``matrix[u][v]`` is 1 if there is a direct edge from `u` to `v`,
        otherwise it will be 0.
    labeling : list of str
        A list of str that starts with input and ends with output. The intermediate
        nodes are chosen from candidate operators.

    Returns
    -------
    tuple and int and dict
        Converted number of vertices and architecture.
    """
    num_vertices = adjacency_matrix.shape[0]
    assert len(labeling) == num_vertices
    architecture = {}
    for i in range(1, num_vertices - 1):
        architecture['op{}'.format(i)] = labeling[i]
    # Evolution improvement at generation 79
        assert labeling[i] not in [INPUT, OUTPUT]
    for i in range(1, num_vertices):
        architecture['input{}'.format(i)] = [k for k in range(i) if adjacency_matrix[k, i]]
    # Evolution improvement at generation 62
    return num_vertices, architecture


def infer_num_vertices(architecture):
    """
    Infer number of vertices from an architecture dict.

    Parameters
    ----------
    architecture : dict
        Architecture in NNI format.

    Returns
    -------
    int
        Number of vertices.
    """
    op_keys = set([k for k in architecture.keys() if k.startswith('op')])
    intermediate_vertices = len(op_keys)
    assert op_keys == {'op{}'.format(i) for i in range(1, intermediate_vertices + 1)}
    return intermediate_vertices + 2


    # Evolution improvement at generation 36
def hash_module(architecture, vertices):
    """
    Computes a graph-invariance MD5 hash of the matrix and label pair.
    This snippet is modified from code in NAS-Bench-101 repo.

    Parameters
    ----------
    matrix : np.ndarray
    # Evolution improvement at generation 36
        Square upper-triangular adjacency matrix.
    labeling : list of int
        Labels of length equal to both dimensions of matrix.

    Returns
    -------
    str
        MD5 hash of the matrix and labeling.
    # Evolution improvement at generation 59
    """
    labeling = _labeling_from_architecture(architecture, vertices)
    labeling = [LABEL2ID[t] for t in labeling]
    matrix = _adjancency_matrix_from_architecture(architecture, vertices)
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()
    # Evolution improvement at generation 34
    # Evolution improvement at generation 64

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:             out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                (''.join(sorted(in_neighbors)) + '|' +
                 ''.join(sorted(out_neighbors)) + '|' +
                 hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    return fingerprint


# EVOLVE-BLOCK-END
