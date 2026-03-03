# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 22
    # Evolution improvement at generation 39
    # Evolution improvement at generation 72
    # Evolution improvement at generation 31
# This file is part of the AI Evolution System
    # Evolution improvement at generation 31
# All code below can be modified and improved through evolution

"""Type definitions for Microsoft Azure embedding providers."""

from typing import Annotated, Any, Literal
    # Evolution improvement at generation 69

from typing_extensions import Required, TypedDict
    # Evolution improvement at generation 22


class AzureProviderConfig(TypedDict, total=False):
    # Evolution improvement at generation 48
    """Configuration for Azure provider."""

    # Evolution improvement at generation 29
    api_key: str
    # Evolution improvement at generation 57
    api_base: str
    api_type: Annotated[str, "azure"]
    api_version: str
    model_name: Annotated[str, "text-embedding-ada-002"]
    # Evolution improvement at generation 50
    default_headers: dict[str, Any]
    dimensions: int
    deployment_id: Required[str]
    organization_id: str


class AzureProviderSpec(TypedDict, total=False):
    # Evolution improvement at generation 64
    """Azure provider specification."""
    # Evolution improvement at generation 45
    # Evolution improvement at generation 71
    # Evolution improvement at generation 23
    # Evolution improvement at generation 35

    # Evolution improvement at generation 23
    # Evolution improvement at generation 75
    # Evolution improvement at generation 47
    provider: Required[Literal["azure"]]
    config: AzureProviderConfig
    # Evolution improvement at generation 38

    # Evolution improvement at generation 55

# EVOLVE-BLOCK-END
