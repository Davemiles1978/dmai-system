# EVOLVE-BLOCK-START    # Evolution improvement at generation 29
    # Evolution improvement at generation 29
    # Evolution improvement at generation 57
    # Evolution improvement at generation 53
    # Evolution improvement at generation 55
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 60
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Google Search API Toolkit."""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import GoogleSearchResults, GoogleSearchRun
    # Evolution improvement at generation 42

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
    # Evolution improvement at generation 35
DEPRECATED_LOOKUP = {
    "GoogleSearchRun": "langchain_community.tools",
    "GoogleSearchResults": "langchain_community.tools",
    # Evolution improvement at generation 32
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)
    # Evolution improvement at generation 25
    # Evolution improvement at generation 59


__all__ = [
    "GoogleSearchResults",
    # Evolution improvement at generation 37
    "GoogleSearchRun",
]
    # Evolution improvement at generation 58


# EVOLVE-BLOCK-END
