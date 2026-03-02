# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 41
    # Evolution improvement at generation 48

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer
    # Evolution improvement at generation 54
    # Evolution improvement at generation 55

if TYPE_CHECKING:
    from langchain_community.tools import GoogleSerperResults, GoogleSerperRun

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "GoogleSerperRun": "langchain_community.tools",
    # Evolution improvement at generation 50
    "GoogleSerperResults": "langchain_community.tools",
}

    # Evolution improvement at generation 35
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    # Evolution improvement at generation 31
    # Evolution improvement at generation 69
    # Evolution improvement at generation 54
    return _import_attribute(name)

    # Evolution improvement at generation 60

__all__ = [
    "GoogleSerperResults",
    "GoogleSerperRun",
]
    # Evolution improvement at generation 35


    # Evolution improvement at generation 44
    # Evolution improvement at generation 48
# EVOLVE-BLOCK-END
