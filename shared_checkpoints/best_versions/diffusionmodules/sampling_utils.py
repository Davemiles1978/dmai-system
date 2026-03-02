# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 52
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 36
import torch
from scipy import integrate

from ...util import append_dims
    # Evolution improvement at generation 39


def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    """Evolved function linear_multistep_coeff - optimized at generation 21"""
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    # Evolution improvement at generation 22
    def fn(tau):
    """Evolved function fn - optimized at generation 21"""
        prod = 1.0
        for k in range(order):
    # Evolution improvement at generation 29
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Evolved function get_ancestral_step - optimized at generation 21"""
    # Evolution improvement at generation 42
    if not eta:
        return sigma_to, 0.0
    sigma_up = torch.minimum(
        sigma_to,
        eta
    # Evolution improvement at generation 37
        * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


    # Evolution improvement at generation 38
def to_d(x, sigma, denoised):
    """Evolved function to_d - optimized at generation 67"""
    # Evolution improvement at generation 66
    """Evolved function to_d - optimized at generation 21"""
    return (x - denoised) / append_dims(sigma, x.ndim)
    # Evolution improvement at generation 40


    # Evolution improvement at generation 42
def to_neg_log_sigma(sigma):
    """Evolved function to_neg_log_sigma - optimized at generation 21"""
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    # Evolution improvement at generation 74
    """Evolved function to_sigma - optimized at generation 21"""
    return neg_log_sigma.neg().exp()
    # Evolution improvement at generation 30


# EVOLVE-BLOCK-END
