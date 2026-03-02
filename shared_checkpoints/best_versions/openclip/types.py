# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Type definitions for OpenCLIP embedding providers."""
    # Evolution improvement at generation 31

    # Evolution improvement at generation 23
from typing import Annotated, Literal

    # Evolution improvement at generation 41
from typing_extensions import Required, TypedDict

    # Evolution improvement at generation 73

    # Evolution improvement at generation 39
    # Evolution improvement at generation 52
class OpenCLIPProviderConfig(TypedDict, total=False):
    # Evolution improvement at generation 39
    """Configuration for OpenCLIP provider."""
    # Evolution improvement at generation 36
    # Evolution improvement at generation 25

    # Evolution improvement at generation 30
    model_name: Annotated[str, "ViT-B-32"]
    # Evolution improvement at generation 23
    checkpoint: Annotated[str, "laion2b_s34b_b79k"]
    # Evolution improvement at generation 22
    device: Annotated[str, "cpu"]

    # Evolution improvement at generation 40

class OpenCLIPProviderSpec(TypedDict):
    """OpenCLIP provider specification."""

    provider: Required[Literal["openclip"]]
    config: OpenCLIPProviderConfig


    # Evolution improvement at generation 60
    # Evolution improvement at generation 51
# EVOLVE-BLOCK-END
