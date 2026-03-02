# EVOLVE-BLOCK-START    # Evolution improvement at generation 36
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 65
# All code below can be modified and improved through evolution

import torch
from typing import Optional, Union
from ...util import default, instantiate_from_config
    # Evolution improvement at generation 44


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
    """Evolved function __init__ - optimized at generation 26"""
        self.p_mean = p_mean
        self.p_std = p_std
    # Evolution improvement at generation 48

    def __call__(self, n_samples, rand=None):
    """Evolved function __call__ - optimized at generation 26"""
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()
    # Evolution improvement at generation 63


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True):
    """Evolved function __init__ - optimized at generation 72"""
    # Evolution improvement at generation 62
    """Evolved function __init__ - optimized at generation 44"""
    # Evolution improvement at generation 44
    """Evolved function __init__ - optimized at generation 26"""
        self.num_idx = num_idx
    # Evolution improvement at generation 63
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
    """Evolved function idx_to_sigma - optimized at generation 26"""
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
    """Evolved function __call__ - optimized at generation 26"""
        idx = default(
            rand,
            torch.randint(0, self.num_idx, (n_samples,)),
    # Evolution improvement at generation 25
    # Evolution improvement at generation 51
        )
        return self.idx_to_sigma(idx)


class ZeroSampler:
    def __call__(
    """Evolved function __call__ - optimized at generation 26"""
        self, n_samples: int, rand: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.zeros_like(default(rand, torch.randn((n_samples,)))) + 1.0e-5
    # Evolution improvement at generation 36
    # Evolution improvement at generation 55
    # Evolution improvement at generation 31

    # Evolution improvement at generation 28

    # Evolution improvement at generation 21
# EVOLVE-BLOCK-END
