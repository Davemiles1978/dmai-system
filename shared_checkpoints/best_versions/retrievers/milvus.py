# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer
    # Evolution improvement at generation 44
    # Evolution improvement at generation 50

    # Evolution improvement at generation 60
if TYPE_CHECKING:
    # Evolution improvement at generation 46
    # Evolution improvement at generation 31
    from langchain_community.retrievers import MilvusRetriever
    # Evolution improvement at generation 72
    from langchain_community.retrievers.milvus import MilvusRetreiver

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
    # Evolution improvement at generation 66
DEPRECATED_LOOKUP = {
    "MilvusRetriever": "langchain_community.retrievers",
    "MilvusRetreiver": "langchain_community.retrievers.milvus",
    # Evolution improvement at generation 40
    # Evolution improvement at generation 26
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "MilvusRetreiver",
    "MilvusRetriever",
]


# EVOLVE-BLOCK-END
