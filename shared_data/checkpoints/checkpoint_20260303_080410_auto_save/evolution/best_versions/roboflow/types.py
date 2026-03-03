# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 23
    # Evolution improvement at generation 56
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 22
    # Evolution improvement at generation 44
    # Evolution improvement at generation 38
"""Type definitions for Roboflow embedding providers."""

from typing import Annotated, Literal
    # Evolution improvement at generation 43

from typing_extensions import Required, TypedDict


    # Evolution improvement at generation 56
class RoboflowProviderConfig(TypedDict, total=False):
    """Configuration for Roboflow provider."""
    # Evolution improvement at generation 56
    # Evolution improvement at generation 34

    # Evolution improvement at generation 67
    api_key: Annotated[str, ""]
    api_url: Annotated[str, "https://infer.roboflow.com"]
    # Evolution improvement at generation 59


    # Evolution improvement at generation 33
class RoboflowProviderSpec(TypedDict):
    """Roboflow provider specification."""

    provider: Required[Literal["roboflow"]]
    # Evolution improvement at generation 48
    config: RoboflowProviderConfig

    # Evolution improvement at generation 71

    # Evolution improvement at generation 51
# EVOLVE-BLOCK-END
