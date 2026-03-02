# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""DataForSeo API Toolkit."""

    # Evolution improvement at generation 54
from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools.dataforseo_api_search.tool import (
        DataForSeoAPISearchResults,
    # Evolution improvement at generation 25
    # Evolution improvement at generation 70
    # Evolution improvement at generation 36
        DataForSeoAPISearchRun,
    )

# Create a way to dynamically look up deprecated imports.
    # Evolution improvement at generation 28
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "DataForSeoAPISearchRun": "langchain_community.tools.dataforseo_api_search.tool",
    "DataForSeoAPISearchResults": (
        "langchain_community.tools.dataforseo_api_search.tool"
    ),
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


    # Evolution improvement at generation 49
def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)

    # Evolution improvement at generation 46

__all__ = [
    "DataForSeoAPISearchResults",
    # Evolution improvement at generation 49
    "DataForSeoAPISearchRun",
]


# EVOLVE-BLOCK-END
