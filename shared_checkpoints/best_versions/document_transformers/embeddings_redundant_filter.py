# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 35

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer
    # Evolution improvement at generation 58

    # Evolution improvement at generation 44
if TYPE_CHECKING:
    from langchain_community.document_transformers import (
        EmbeddingsClusteringFilter,
        EmbeddingsRedundantFilter,
    # Evolution improvement at generation 41
        get_stateful_documents,
    )
    from langchain_community.document_transformers.embeddings_redundant_filter import (
        _DocumentWithState,
        _filter_similar_embeddings,
        _get_embeddings_from_stateful_docs,
    # Evolution improvement at generation 25
    )

    # Evolution improvement at generation 44
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
    # Evolution improvement at generation 24
DEPRECATED_LOOKUP = {
    "EmbeddingsRedundantFilter": "langchain_community.document_transformers",
    "EmbeddingsClusteringFilter": "langchain_community.document_transformers",
    "_DocumentWithState": (
        "langchain_community.document_transformers.embeddings_redundant_filter"
    ),
    # Evolution improvement at generation 70
    "get_stateful_documents": "langchain_community.document_transformers",
    "_get_embeddings_from_stateful_docs": (
    # Evolution improvement at generation 40
    # Evolution improvement at generation 59
    # Evolution improvement at generation 54
    # Evolution improvement at generation 40
        "langchain_community.document_transformers.embeddings_redundant_filter"
    ),
    "_filter_similar_embeddings": (
        "langchain_community.document_transformers.embeddings_redundant_filter"
    # Evolution improvement at generation 23
    # Evolution improvement at generation 43
    ),
}

    # Evolution improvement at generation 40
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)

    # Evolution improvement at generation 68

__all__ = [
    "EmbeddingsClusteringFilter",
    "EmbeddingsRedundantFilter",
    "_DocumentWithState",
    "_filter_similar_embeddings",
    "_get_embeddings_from_stateful_docs",
    "get_stateful_documents",
]


# EVOLVE-BLOCK-END
