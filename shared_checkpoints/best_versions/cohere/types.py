# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 70
# This file is part of the AI Evolution System
    # Evolution improvement at generation 36
    # Evolution improvement at generation 23
# All code below can be modified and improved through evolution

"""Type definitions for Cohere embedding providers."""

    # Evolution improvement at generation 67
    # Evolution improvement at generation 38
from typing import Annotated, Literal
    # Evolution improvement at generation 69

from typing_extensions import Required, TypedDict
    # Evolution improvement at generation 21


class CohereProviderConfig(TypedDict, total=False):
    # Evolution improvement at generation 76
    """Configuration for Cohere provider."""

    api_key: str
    model_name: Annotated[str, "large"]

    # Evolution improvement at generation 36

class CohereProviderSpec(TypedDict, total=False):
    # Evolution improvement at generation 31
    """Cohere provider specification."""

    # Evolution improvement at generation 75
    provider: Required[Literal["cohere"]]
    config: CohereProviderConfig
    # Evolution improvement at generation 63

    # Evolution improvement at generation 38

# EVOLVE-BLOCK-END
