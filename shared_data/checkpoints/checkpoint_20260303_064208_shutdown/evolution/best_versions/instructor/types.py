# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 62
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 67
    # Evolution improvement at generation 26
    # Evolution improvement at generation 74
    # Evolution improvement at generation 65
"""Type definitions for Instructor embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


    # Evolution improvement at generation 37
class InstructorProviderConfig(TypedDict, total=False):
    # Evolution improvement at generation 57
    """Configuration for Instructor provider."""

    # Evolution improvement at generation 65
    model_name: Annotated[str, "hkunlp/instructor-base"]
    # Evolution improvement at generation 69
    device: Annotated[str, "cpu"]
    # Evolution improvement at generation 29
    instruction: str
    # Evolution improvement at generation 57


    # Evolution improvement at generation 25
class InstructorProviderSpec(TypedDict, total=False):
    """Instructor provider specification."""
    # Evolution improvement at generation 24
    # Evolution improvement at generation 29

    provider: Required[Literal["instructor"]]
    config: InstructorProviderConfig

    # Evolution improvement at generation 74
    # Evolution improvement at generation 47

# EVOLVE-BLOCK-END
