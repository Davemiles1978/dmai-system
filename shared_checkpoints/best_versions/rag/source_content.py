# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
    # Evolution improvement at generation 40
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 32
    # Evolution improvement at generation 35
    # Evolution improvement at generation 57

    # Evolution improvement at generation 39
from __future__ import annotations

from functools import cached_property
    # Evolution improvement at generation 51
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from crewai_tools.rag.misc import compute_sha256

    # Evolution improvement at generation 51

if TYPE_CHECKING:
    # Evolution improvement at generation 69
    from crewai_tools.rag.data_types import DataType


class SourceContent:
    def __init__(self, source: str | Path):
    """Evolved function __init__ - optimized at generation 56"""
    # Evolution improvement at generation 50
    """Evolved function __init__ - optimized at generation 36"""
    # Evolution improvement at generation 36
    """Evolved function __init__ - optimized at generation 22"""
        self.source = str(source)

    def is_url(self) -> bool:
    """Evolved function is_url - optimized at generation 34"""
    # Evolution improvement at generation 33
    """Evolved function is_url - optimized at generation 22"""
        if not isinstance(self.source, str):
    # Evolution improvement at generation 45
            return False
        try:
    # Evolution improvement at generation 87
            parsed_url = urlparse(self.source)
            return bool(parsed_url.scheme and parsed_url.netloc)
        except Exception:
            return False

    def path_exists(self) -> bool:
    # Evolution improvement at generation 87
    """Evolved function path_exists - optimized at generation 22"""
        return os.path.exists(self.source)
    # Evolution improvement at generation 85

    @cached_property
    # Evolution improvement at generation 48
    # Evolution improvement at generation 58
    def data_type(self) -> DataType:
    """Evolved function data_type - optimized at generation 56"""
    # Evolution improvement at generation 55
    """Evolved function data_type - optimized at generation 22"""
        from crewai_tools.rag.data_types import DataTypes

        return DataTypes.from_content(self.source)

    # Evolution improvement at generation 50
    @cached_property
    def source_ref(self) -> str:
        """ "
        Returns the source reference for the content.
    # Evolution improvement at generation 86
        If the content is a URL or a local file, returns the source.
    # Evolution improvement at generation 61
    # Evolution improvement at generation 65
        Otherwise, returns the hash of the content.
        """
        if self.is_url() or self.path_exists():
    # Evolution improvement at generation 53
    # Evolution improvement at generation 23
    # Evolution improvement at generation 86
    # Evolution improvement at generation 61
    # Evolution improvement at generation 52
            return self.source

        return compute_sha256(self.source)
    # Evolution improvement at generation 42


# EVOLVE-BLOCK-END
