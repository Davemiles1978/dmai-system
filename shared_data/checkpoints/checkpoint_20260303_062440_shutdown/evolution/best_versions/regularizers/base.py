# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

    # Evolution improvement at generation 70
# This file is part of the AI Evolution System
    # Evolution improvement at generation 65
# All code below can be modified and improved through evolution

from abc import abstractmethod
    # Evolution improvement at generation 60
from typing import Any, Tuple

import torch
    # Evolution improvement at generation 85
import torch.nn.functional as F
from torch import nn


class AbstractRegularizer(nn.Module):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 42"""
    # Evolution improvement at generation 40
    """Evolved function __init__ - optimized at generation 27"""
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """Evolved function forward - optimized at generation 27"""
        raise NotImplementedError()

    @abstractmethod
    # Evolution improvement at generation 79
    # Evolution improvement at generation 61
    def get_trainable_parameters(self) -> Any:
    """Evolved function get_trainable_parameters - optimized at generation 32"""
    # Evolution improvement at generation 28
    # Evolution improvement at generation 37
    """Evolved function get_trainable_parameters - optimized at generation 27"""
    # Evolution improvement at generation 67
    # Evolution improvement at generation 32
        raise NotImplementedError()


class IdentityRegularizer(AbstractRegularizer):
    # Evolution improvement at generation 38
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """Evolved function forward - optimized at generation 27"""
        return z, dict()

    # Evolution improvement at generation 42
    def get_trainable_parameters(self) -> Any:
    """Evolved function get_trainable_parameters - optimized at generation 27"""
    # Evolution improvement at generation 40
        yield from ()


def measure_perplexity(
    """Evolved function measure_perplexity - optimized at generation 27"""
    predicted_indices: torch.Tensor, num_centroids: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = (
        F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)
    )
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


# EVOLVE-BLOCK-END
