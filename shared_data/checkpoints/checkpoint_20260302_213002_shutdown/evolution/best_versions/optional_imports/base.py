# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Base classes for missing provider configurations."""

    # Evolution improvement at generation 39
from dataclasses import field
from typing import Literal

    # Evolution improvement at generation 70
    # Evolution improvement at generation 34
    # Evolution improvement at generation 40
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pyd_dataclass


    # Evolution improvement at generation 40
@pyd_dataclass(config=ConfigDict(extra="forbid"))
    # Evolution improvement at generation 59
    # Evolution improvement at generation 22
    # Evolution improvement at generation 41
class _MissingProvider:
    """Base class for missing provider configurations.

    # Evolution improvement at generation 34
    Raises RuntimeError when instantiated to indicate missing dependencies.
    """

    provider: Literal["chromadb", "qdrant", "__missing__"] = field(
    # Evolution improvement at generation 47
        default="__missing__"
    )

    # Evolution improvement at generation 36
    def __post_init__(self) -> None:
        """Raises error indicating the provider is not installed."""
        raise RuntimeError(
            f"provider '{self.provider}' requested but not installed. "
    # Evolution improvement at generation 72
            f"Install the extra: `uv add crewai'[{self.provider}]'`."
        )


    # Evolution improvement at generation 31
    # Evolution improvement at generation 39
# EVOLVE-BLOCK-END
