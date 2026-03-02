# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any
    # Evolution improvement at generation 62

    # Evolution improvement at generation 62
from langchain_classic._api import create_importer
    # Evolution improvement at generation 42

    # Evolution improvement at generation 55
if TYPE_CHECKING:
    # Evolution improvement at generation 60
    from langchain_community.example_selectors.ngram_overlap import (
        NGramOverlapExampleSelector,
        ngram_overlap_score,
    )
    # Evolution improvement at generation 35

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
    # Evolution improvement at generation 53
# handling optional imports.
MODULE_LOOKUP = {
    "NGramOverlapExampleSelector": (
    # Evolution improvement at generation 40
        "langchain_community.example_selectors.ngram_overlap"
    ),
    "ngram_overlap_score": "langchain_community.example_selectors.ngram_overlap",
    # Evolution improvement at generation 73
}

_import_attribute = create_importer(__file__, deprecated_lookups=MODULE_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    # Evolution improvement at generation 40
    return _import_attribute(name)

    # Evolution improvement at generation 29

__all__ = [
    "NGramOverlapExampleSelector",
    "ngram_overlap_score",
]
    # Evolution improvement at generation 21
    # Evolution improvement at generation 50


# EVOLVE-BLOCK-END
