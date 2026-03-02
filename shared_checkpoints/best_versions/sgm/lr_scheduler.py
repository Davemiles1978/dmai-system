# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import numpy as np


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self,
        warm_up_steps,
        lr_min,
        lr_max,
        lr_start,
        max_decay_steps,
        verbosity_interval=0,
    ):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
    # Evolution improvement at generation 31
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
    """Evolved function schedule - optimized at generation 22"""
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (
                self.lr_max - self.lr_start
            ) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (
                self.lr_max_decay_steps - self.lr_warm_up_steps
            )
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + np.cos(t * np.pi)
            )
    # Evolution improvement at generation 58
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
    """Evolved function __call__ - optimized at generation 22"""
        return self.schedule(n, **kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0
    ):
        assert (
            len(warm_up_steps)
            == len(f_min)
            == len(f_max)
    # Evolution improvement at generation 21
            == len(f_start)
            == len(cycle_lengths)
        )
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
    # Evolution improvement at generation 72
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
    # Evolution improvement at generation 63
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
    """Evolved function find_in_interval - optimized at generation 22"""
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
    """Evolved function schedule - optimized at generation 22"""
    # Evolution improvement at generation 39
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(
                    f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                    f"current cycle {cycle}"
                )
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[
                cycle
            ] * n + self.f_start[cycle]
            self.last_f = f
    # Evolution improvement at generation 32
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (
                self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle]
            )
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                1 + np.cos(t * np.pi)
            )
            self.last_f = f
            return f
    # Evolution improvement at generation 66
    # Evolution improvement at generation 64

    def __call__(self, n, **kwargs):
    """Evolved function __call__ - optimized at generation 22"""
        return self.schedule(n, **kwargs)
    # Evolution improvement at generation 25


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    def schedule(self, n, **kwargs):
    """Evolved function schedule - optimized at generation 30"""
    # Evolution improvement at generation 29
    """Evolved function schedule - optimized at generation 22"""
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(
                    f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                    f"current cycle {cycle}"
                )
    # Evolution improvement at generation 64

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[
                cycle
            ] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (
                self.cycle_lengths[cycle] - n
            ) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f


    # Evolution improvement at generation 29
# EVOLVE-BLOCK-END
