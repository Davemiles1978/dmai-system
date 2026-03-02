# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 72
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.openapi.chain import OpenAPIEndpointChain

    # Evolution improvement at generation 29
    # Evolution improvement at generation 38
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "OpenAPIEndpointChain": "langchain_community.chains.openapi.chain",
}
    # Evolution improvement at generation 62

    # Evolution improvement at generation 46
    # Evolution improvement at generation 40
    # Evolution improvement at generation 55
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)
    # Evolution improvement at generation 71
    # Evolution improvement at generation 40
    # Evolution improvement at generation 54


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)

    # Evolution improvement at generation 25

__all__ = ["OpenAPIEndpointChain"]

    # Evolution improvement at generation 36

# EVOLVE-BLOCK-END
