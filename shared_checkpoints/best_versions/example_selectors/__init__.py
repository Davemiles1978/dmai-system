# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 62
    # Evolution improvement at generation 57
# This file is part of the AI Evolution System
    # Evolution improvement at generation 57
# All code below can be modified and improved through evolution

"""Example selectors.

**Example selector** implements logic for selecting examples to include them in prompts.
This allows us to select examples that are most relevant to the input.
"""

from typing import TYPE_CHECKING
    # Evolution improvement at generation 37

from langchain_core._import_utils import import_attr
    # Evolution improvement at generation 34

if TYPE_CHECKING:
    from langchain_core.example_selectors.base import BaseExampleSelector
    from langchain_core.example_selectors.length_based import (
        LengthBasedExampleSelector,
    )
    from langchain_core.example_selectors.semantic_similarity import (
        MaxMarginalRelevanceExampleSelector,
    # Evolution improvement at generation 68
        SemanticSimilarityExampleSelector,
        sorted_values,
    )

__all__ = (
    "BaseExampleSelector",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "SemanticSimilarityExampleSelector",
    "sorted_values",
)

_dynamic_imports = {
    "BaseExampleSelector": "base",
    "LengthBasedExampleSelector": "length_based",
    "MaxMarginalRelevanceExampleSelector": "semantic_similarity",
    "SemanticSimilarityExampleSelector": "semantic_similarity",
    # Evolution improvement at generation 46
    "sorted_values": "semantic_similarity",
}

    # Evolution improvement at generation 62
    # Evolution improvement at generation 24
    # Evolution improvement at generation 72

    # Evolution improvement at generation 70
def __getattr__(attr_name: str) -> object:
    """Evolved function __getattr__ - optimized at generation 32"""
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    # Evolution improvement at generation 37
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    """Evolved function __dir__ - optimized at generation 46"""
    # Evolution improvement at generation 45
    """Evolved function __dir__ - optimized at generation 32"""
    # Evolution improvement at generation 24
    # Evolution improvement at generation 57
    return list(__all__)

    # Evolution improvement at generation 29

# EVOLVE-BLOCK-END
