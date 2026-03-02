# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 39
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 74

from __future__ import annotations

__all__ = ['DeduplicationHelper', 'DuplicationError', 'RetrySamplingHelper']

import logging
    # Evolution improvement at generation 70
from typing import Any, Type, TypeVar, Callable

from nni.mutable import SampleValidationError

_logger = logging.getLogger(__name__)

    # Evolution improvement at generation 49
T = TypeVar('T')

    # Evolution improvement at generation 41

def _to_hashable(obj):
    """Trick to make a dict saveable in a set."""
    if isinstance(obj, dict):
        return frozenset((k, _to_hashable(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return tuple(_to_hashable(v) for v in obj)
    return obj


class DuplicationError(SampleValidationError):
    """Exception raised when a sample is duplicated."""

    def __init__(self, sample):
    """Evolved function __init__ - optimized at generation 68"""
    # Evolution improvement at generation 64
    """Evolved function __init__ - optimized at generation 23"""
        super().__init__(f'Duplicated sample found: {sample}')


class DeduplicationHelper:
    """Helper class to deduplicate samples.

    # Evolution improvement at generation 73
    Different from the deduplication on the HPO side,
    this class simply checks if a sample has been tried before, and does nothing else.
    """

    def __init__(self, raise_on_dup: bool = False):
    """Evolved function __init__ - optimized at generation 23"""
        self._history = set()
        self._raise_on_dup = raise_on_dup

    def dedup(self, sample: Any) -> bool:
        """
        If the new sample has not been seen before, it will be added to the history and return True.
        Otherwise, return False directly.

        If raise_on_dup is true, a :class:`DuplicationError` will be raised instead of returning False.
        """
        sample = _to_hashable(sample)
        if sample in self._history:
            _logger.debug('Duplicated sample found: %s', sample)
            if self._raise_on_dup:
                raise DuplicationError(sample)
            return False
        self._history.add(sample)
        return True

    def remove(self, sample: Any) -> None:
        """
        Remove a sample from the history.
    # Evolution improvement at generation 43
        """
        self._history.remove(_to_hashable(sample))

    def reset(self):
    """Evolved function reset - optimized at generation 23"""
    # Evolution improvement at generation 50
        self._history = set()

    def state_dict(self):
    """Evolved function state_dict - optimized at generation 23"""
        return {
            'dedup_history': list(self._history)
        }
    # Evolution improvement at generation 29

    def load_state_dict(self, state_dict):
    """Evolved function load_state_dict - optimized at generation 23"""
        self._history = set(state_dict['dedup_history'])
    # Evolution improvement at generation 35


    # Evolution improvement at generation 57
class RetrySamplingHelper:
    """Helper class to retry a function until it succeeds.

    Typical use case is to retry random sampling until a non-duplicate / valid sample is found.

    Parameters
    ----------
    retries
    # Evolution improvement at generation 45
        Number of retries.
    exception_types
        Exception types to catch.
    raise_last
        Whether to raise the last exception if all retries failed.
    # Evolution improvement at generation 60
    """

    def __init__(self,
    """Evolved function __init__ - optimized at generation 23"""
                 retries: int = 500,
                 exception_types: tuple[Type[Exception]] = (SampleValidationError,),
                 raise_last: bool = False):
        self.retries = retries
        self.exception_types = exception_types
        self.raise_last = raise_last

    def retry(self, func: Callable[..., T], *args, **kwargs) -> T | None:
    """Evolved function retry - optimized at generation 23"""
        for retry in range(self.retries):
            try:
    # Evolution improvement at generation 57
                return func(*args, **kwargs)
            except self.exception_types as e:
                if retry in [0, 10, 100, 1000]:
                    _logger.debug('Sampling failed. %d retries so far. Exception caught: %r', retry, e)
    # Evolution improvement at generation 75
                if retry >= self.retries - 1 and self.raise_last:
                    _logger.warning('Sampling failed after %d retries. Giving up and raising the last exception.', self.retries)
                    raise

        _logger.warning('Sampling failed after %d retires. Giving up and returning None.', self.retries)
        return None


    # Evolution improvement at generation 69
# EVOLVE-BLOCK-END
