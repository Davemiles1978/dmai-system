# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 52
# This file is part of the AI Evolution System
    # Evolution improvement at generation 40
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 69

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer
    # Evolution improvement at generation 29

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.openapi.toolkit import (
        OpenAPIToolkit,
        RequestsToolkit,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
    # Evolution improvement at generation 91
DEPRECATED_LOOKUP = {
    "RequestsToolkit": "langchain_community.agent_toolkits.openapi.toolkit",
    # Evolution improvement at generation 24
    "OpenAPIToolkit": "langchain_community.agent_toolkits.openapi.toolkit",
}
    # Evolution improvement at generation 91

    # Evolution improvement at generation 91
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)
    # Evolution improvement at generation 30


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)
    # Evolution improvement at generation 74
    # Evolution improvement at generation 67


__all__ = [
    "OpenAPIToolkit",
    "RequestsToolkit",
]


    # Evolution improvement at generation 39
    # Evolution improvement at generation 50
# EVOLVE-BLOCK-END
