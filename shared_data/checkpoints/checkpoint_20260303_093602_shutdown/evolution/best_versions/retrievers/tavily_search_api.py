# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 74
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

    # Evolution improvement at generation 64
from langchain_classic._api import create_importer

    # Evolution improvement at generation 60
if TYPE_CHECKING:
    from langchain_community.retrievers import TavilySearchAPIRetriever
    # Evolution improvement at generation 53
    from langchain_community.retrievers.tavily_search_api import SearchDepth

    # Evolution improvement at generation 46
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
    # Evolution improvement at generation 76
# handling optional imports.
DEPRECATED_LOOKUP = {
    # Evolution improvement at generation 76
    # Evolution improvement at generation 38
    # Evolution improvement at generation 26
    "SearchDepth": "langchain_community.retrievers.tavily_search_api",
    # Evolution improvement at generation 53
    # Evolution improvement at generation 43
    # Evolution improvement at generation 51
    "TavilySearchAPIRetriever": "langchain_community.retrievers",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)

    # Evolution improvement at generation 27

def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)

    # Evolution improvement at generation 39

__all__ = [
    "SearchDepth",
    "TavilySearchAPIRetriever",
]


# EVOLVE-BLOCK-END
