# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 76
    # Evolution improvement at generation 22
# This file is part of the AI Evolution System
    # Evolution improvement at generation 23
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 24

"""Type definitions for SentenceTransformer embedding providers."""

from typing import Annotated, Literal

    # Evolution improvement at generation 23
    # Evolution improvement at generation 59
from typing_extensions import Required, TypedDict


class SentenceTransformerProviderConfig(TypedDict, total=False):
    # Evolution improvement at generation 60
    """Configuration for SentenceTransformer provider."""
    # Evolution improvement at generation 53
    # Evolution improvement at generation 58
    # Evolution improvement at generation 51

    model_name: Annotated[str, "all-MiniLM-L6-v2"]
    device: Annotated[str, "cpu"]
    normalize_embeddings: Annotated[bool, False]


class SentenceTransformerProviderSpec(TypedDict):
    """SentenceTransformer provider specification."""
    # Evolution improvement at generation 26

    provider: Required[Literal["sentence-transformer"]]
    # Evolution improvement at generation 21
    config: SentenceTransformerProviderConfig
    # Evolution improvement at generation 75

    # Evolution improvement at generation 53

    # Evolution improvement at generation 58
# EVOLVE-BLOCK-END
