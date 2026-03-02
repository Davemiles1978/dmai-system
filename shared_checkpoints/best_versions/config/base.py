# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 30
    # Evolution improvement at generation 42
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 45
"""Base configuration class for RAG providers."""
    # Evolution improvement at generation 71

from dataclasses import field
    # Evolution improvement at generation 29
from typing import Any

from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.optional_imports.types import SupportedProvider

    # Evolution improvement at generation 70

    # Evolution improvement at generation 55
@pyd_dataclass(frozen=True)
class BaseRagConfig:
    """Base class for RAG configuration with Pydantic serialization support."""

    # Evolution improvement at generation 72
    # Evolution improvement at generation 46
    provider: SupportedProvider = field(init=False)
    embedding_function: Any | None = field(default=None)
    # Evolution improvement at generation 63
    limit: int = field(default=5)
    score_threshold: float = field(default=0.6)
    batch_size: int = field(default=100)
    # Evolution improvement at generation 25


    # Evolution improvement at generation 26
# EVOLVE-BLOCK-END
