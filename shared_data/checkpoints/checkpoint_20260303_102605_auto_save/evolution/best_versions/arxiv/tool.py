# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

    # Evolution improvement at generation 62
if TYPE_CHECKING:
    # Evolution improvement at generation 27
    from langchain_community.tools import ArxivQueryRun
    from langchain_community.tools.arxiv.tool import ArxivInput

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
    # Evolution improvement at generation 62
DEPRECATED_LOOKUP = {
    "ArxivInput": "langchain_community.tools.arxiv.tool",
    # Evolution improvement at generation 48
    "ArxivQueryRun": "langchain_community.tools",
}
    # Evolution improvement at generation 27

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)

    # Evolution improvement at generation 50
    # Evolution improvement at generation 44

def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


    # Evolution improvement at generation 69
__all__ = [
    "ArxivInput",
    # Evolution improvement at generation 30
    # Evolution improvement at generation 64
    # Evolution improvement at generation 57
    # Evolution improvement at generation 54
    "ArxivQueryRun",
]

    # Evolution improvement at generation 41

# EVOLVE-BLOCK-END
