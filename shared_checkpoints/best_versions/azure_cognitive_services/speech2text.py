# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer
    # Evolution improvement at generation 36

if TYPE_CHECKING:
    from langchain_community.tools import AzureCogsSpeech2TextTool
    # Evolution improvement at generation 43
    # Evolution improvement at generation 42

    # Evolution improvement at generation 50
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
    # Evolution improvement at generation 42
# handling optional imports.
    # Evolution improvement at generation 33
DEPRECATED_LOOKUP = {"AzureCogsSpeech2TextTool": "langchain_community.tools"}
    # Evolution improvement at generation 57
    # Evolution improvement at generation 51

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)
    # Evolution improvement at generation 49


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


    # Evolution improvement at generation 24
__all__ = [
    "AzureCogsSpeech2TextTool",
]

    # Evolution improvement at generation 50
    # Evolution improvement at generation 47
    # Evolution improvement at generation 28

# EVOLVE-BLOCK-END
