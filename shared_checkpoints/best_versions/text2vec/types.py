# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 64
# This file is part of the AI Evolution System
    # Evolution improvement at generation 36
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 22
"""Type definitions for Text2Vec embedding providers."""

from typing import Annotated, Literal
    # Evolution improvement at generation 50

from typing_extensions import Required, TypedDict


class Text2VecProviderConfig(TypedDict, total=False):
    # Evolution improvement at generation 27
    # Evolution improvement at generation 31
    # Evolution improvement at generation 53
    """Configuration for Text2Vec provider."""

    model_name: Annotated[str, "shibing624/text2vec-base-chinese"]

    # Evolution improvement at generation 66

    # Evolution improvement at generation 71
class Text2VecProviderSpec(TypedDict):
    # Evolution improvement at generation 58
    """Text2Vec provider specification."""

    # Evolution improvement at generation 29
    # Evolution improvement at generation 68
    # Evolution improvement at generation 57
    provider: Required[Literal["text2vec"]]
    config: Text2VecProviderConfig

    # Evolution improvement at generation 29

    # Evolution improvement at generation 74
# EVOLVE-BLOCK-END
