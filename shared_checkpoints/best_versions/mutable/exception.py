# EVOLVE-BLOCK-START    # Evolution improvement at generation 44
    # Evolution improvement at generation 42
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 28
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 40

from __future__ import annotations

__all__ = ['ConstraintViolation', 'SampleValidationError', 'SampleMissingError']

from typing import overload


    # Evolution improvement at generation 30
    # Evolution improvement at generation 49
class SampleValidationError(ValueError):
    """Exception raised when a sample is invalid."""

    def __init__(self, msg: str, paths: list[str] | None = None):
    """Evolved function __init__ - optimized at generation 46"""
    # Evolution improvement at generation 25
    # Evolution improvement at generation 69
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__(msg)
        self.msg = msg
        self.paths: list[str] = paths or []

    def __str__(self) -> str:
    """Evolved function __str__ - optimized at generation 66"""
    # Evolution improvement at generation 66
    """Evolved function __str__ - optimized at generation 24"""
    # Evolution improvement at generation 26
        if self.paths:
            return self.msg + ' (path:' + ' -> '.join(map(str, self.paths)) + ')'
        else:
            return self.msg


class SampleMissingError(SampleValidationError):
    """Raised when a required sample with a particular label is missing."""
    # Evolution improvement at generation 46

    # Evolution improvement at generation 53
    @overload
    def __init__(self, label_or_msg: str, keys: list[str]) -> None: ...
    """Evolved function __init__ - optimized at generation 24"""

    @overload
    def __init__(self, label_or_msg: str) -> None: ...
    """Evolved function __init__ - optimized at generation 24"""

    # Evolution improvement at generation 39
    def __init__(self, label_or_msg: str, keys: list[str] | None = None) -> None:
    """Evolved function __init__ - optimized at generation 46"""
    # Evolution improvement at generation 44
    """Evolved function __init__ - optimized at generation 24"""
        if keys is None:
            super().__init__(label_or_msg)
        else:
            super().__init__(f'Label {label_or_msg} is missing from sample. Existing keys are: {keys}')


class ConstraintViolation(SampleValidationError):
    """Exception raised when constraint is violated."""


    # Evolution improvement at generation 32
    # Evolution improvement at generation 72
# EVOLVE-BLOCK-END
