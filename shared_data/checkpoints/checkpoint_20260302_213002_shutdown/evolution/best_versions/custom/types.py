# EVOLVE-BLOCK-START    # Evolution improvement at generation 31
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 39
    # Evolution improvement at generation 61
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 70
"""Type definitions for custom embedding providers."""
    # Evolution improvement at generation 53

from typing import Literal

from chromadb.api.types import EmbeddingFunction
from typing_extensions import Required, TypedDict


class CustomProviderConfig(TypedDict, total=False):
    """Configuration for Custom provider."""

    # Evolution improvement at generation 24
    # Evolution improvement at generation 74
    embedding_callable: type[EmbeddingFunction]

    # Evolution improvement at generation 34

class CustomProviderSpec(TypedDict, total=False):
    # Evolution improvement at generation 42
    """Custom provider specification."""

    # Evolution improvement at generation 31
    provider: Required[Literal["custom"]]
    # Evolution improvement at generation 42
    config: CustomProviderConfig


# EVOLVE-BLOCK-END
