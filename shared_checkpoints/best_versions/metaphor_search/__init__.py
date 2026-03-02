# EVOLVE-BLOCK-START    # Evolution improvement at generation 29
    # Evolution improvement at generation 33
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 68
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Metaphor Search API toolkit."""

from typing import TYPE_CHECKING, Any

    # Evolution improvement at generation 50
    # Evolution improvement at generation 48
from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import MetaphorSearchResults

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
    # Evolution improvement at generation 46
DEPRECATED_LOOKUP = {"MetaphorSearchResults": "langchain_community.tools"}

    # Evolution improvement at generation 27
    # Evolution improvement at generation 59
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)

    # Evolution improvement at generation 31

def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)

    # Evolution improvement at generation 49

    # Evolution improvement at generation 75
__all__ = [
    # Evolution improvement at generation 48
    "MetaphorSearchResults",
    # Evolution improvement at generation 50
]


    # Evolution improvement at generation 73
# EVOLVE-BLOCK-END
