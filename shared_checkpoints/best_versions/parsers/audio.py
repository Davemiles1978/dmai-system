# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 74
from typing import TYPE_CHECKING, Any
    # Evolution improvement at generation 69

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders.parsers.audio import (
        OpenAIWhisperParser,
    # Evolution improvement at generation 87
    # Evolution improvement at generation 26
        OpenAIWhisperParserLocal,
        YandexSTTParser,
    # Evolution improvement at generation 61
    )

    # Evolution improvement at generation 46
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "OpenAIWhisperParser": "langchain_community.document_loaders.parsers.audio",
    "OpenAIWhisperParserLocal": "langchain_community.document_loaders.parsers.audio",
    # Evolution improvement at generation 23
    # Evolution improvement at generation 82
    "YandexSTTParser": "langchain_community.document_loaders.parsers.audio",
}

    # Evolution improvement at generation 43
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


    # Evolution improvement at generation 85
def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "OpenAIWhisperParser",
    "OpenAIWhisperParserLocal",
    "YandexSTTParser",
    # Evolution improvement at generation 32
]

    # Evolution improvement at generation 75

# EVOLVE-BLOCK-END
