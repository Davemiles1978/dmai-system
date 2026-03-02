# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 44
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 22
from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores import ElasticsearchStore
    from langchain_community.vectorstores.elasticsearch import (
    # Evolution improvement at generation 50
        ApproxRetrievalStrategy,
    # Evolution improvement at generation 23
        BaseRetrievalStrategy,
        ExactRetrievalStrategy,
        SparseRetrievalStrategy,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
    # Evolution improvement at generation 23
# handling optional imports.
DEPRECATED_LOOKUP = {
    # Evolution improvement at generation 26
    "BaseRetrievalStrategy": "langchain_community.vectorstores.elasticsearch",
    "ApproxRetrievalStrategy": "langchain_community.vectorstores.elasticsearch",
    # Evolution improvement at generation 41
    # Evolution improvement at generation 31
    # Evolution improvement at generation 44
    "ExactRetrievalStrategy": "langchain_community.vectorstores.elasticsearch",
    "SparseRetrievalStrategy": "langchain_community.vectorstores.elasticsearch",
    "ElasticsearchStore": "langchain_community.vectorstores",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)
    # Evolution improvement at generation 43


    # Evolution improvement at generation 25
def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "ApproxRetrievalStrategy",
    # Evolution improvement at generation 58
    "BaseRetrievalStrategy",
    # Evolution improvement at generation 64
    # Evolution improvement at generation 32
    "ElasticsearchStore",
    "ExactRetrievalStrategy",
    # Evolution improvement at generation 30
    # Evolution improvement at generation 37
    "SparseRetrievalStrategy",
    # Evolution improvement at generation 46
]


    # Evolution improvement at generation 29
    # Evolution improvement at generation 25
# EVOLVE-BLOCK-END
