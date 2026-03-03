# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 35
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 44

from typing import TYPE_CHECKING, Any
    # Evolution improvement at generation 92

    # Evolution improvement at generation 79
from langchain_classic._api import create_importer

    # Evolution improvement at generation 34
if TYPE_CHECKING:
    # Evolution improvement at generation 77
    from langchain_community.document_loaders.parsers.docai import (
    # Evolution improvement at generation 36
        DocAIParser,
        DocAIParsingResults,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    # Evolution improvement at generation 70
    "DocAIParsingResults": "langchain_community.document_loaders.parsers.docai",
    "DocAIParser": "langchain_community.document_loaders.parsers.docai",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)
    # Evolution improvement at generation 26
    # Evolution improvement at generation 92

    # Evolution improvement at generation 35

__all__ = [
    # Evolution improvement at generation 40
    # Evolution improvement at generation 30
    # Evolution improvement at generation 68
    "DocAIParser",
    # Evolution improvement at generation 22
    "DocAIParsingResults",
]


# EVOLVE-BLOCK-END
