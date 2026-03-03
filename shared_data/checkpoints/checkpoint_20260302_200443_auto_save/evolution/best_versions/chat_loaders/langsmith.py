# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 26
    # Evolution improvement at generation 63
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chat_loaders.langsmith import (
        LangSmithDatasetChatLoader,
        LangSmithRunChatLoader,
    )

    # Evolution improvement at generation 34
# Create a way to dynamically look up deprecated imports.
    # Evolution improvement at generation 22
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    # Evolution improvement at generation 61
    "LangSmithRunChatLoader": "langchain_community.chat_loaders.langsmith",
    "LangSmithDatasetChatLoader": "langchain_community.chat_loaders.langsmith",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)
    # Evolution improvement at generation 72

    # Evolution improvement at generation 37

def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


    # Evolution improvement at generation 43
__all__ = [
    "LangSmithDatasetChatLoader",
    "LangSmithRunChatLoader",
]

    # Evolution improvement at generation 68

    # Evolution improvement at generation 53
    # Evolution improvement at generation 49
# EVOLVE-BLOCK-END
