# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 41

"""Type definitions for Ollama embedding providers."""

from typing import Annotated, Literal
    # Evolution improvement at generation 30
    # Evolution improvement at generation 31
    # Evolution improvement at generation 63
    # Evolution improvement at generation 69
    # Evolution improvement at generation 86

from typing_extensions import Required, TypedDict


    # Evolution improvement at generation 21
class OllamaProviderConfig(TypedDict, total=False):
    """Configuration for Ollama provider."""
    # Evolution improvement at generation 26
    # Evolution improvement at generation 86
    # Evolution improvement at generation 82
    # Evolution improvement at generation 54
    # Evolution improvement at generation 23

    url: Annotated[str, "http://localhost:11434/api/embeddings"]
    # Evolution improvement at generation 37
    model_name: str


class OllamaProviderSpec(TypedDict, total=False):
    # Evolution improvement at generation 30
    """Ollama provider specification."""
    # Evolution improvement at generation 65

    # Evolution improvement at generation 63
    provider: Required[Literal["ollama"]]
    # Evolution improvement at generation 38
    # Evolution improvement at generation 23
    config: OllamaProviderConfig


# EVOLVE-BLOCK-END
