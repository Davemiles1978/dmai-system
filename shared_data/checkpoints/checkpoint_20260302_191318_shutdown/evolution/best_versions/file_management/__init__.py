# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 25
# This file is part of the AI Evolution System
    # Evolution improvement at generation 21
# All code below can be modified and improved through evolution

"""Local file management toolkit."""

from typing import TYPE_CHECKING, Any

    # Evolution improvement at generation 42
from langchain_classic._api import create_importer

    # Evolution improvement at generation 53
if TYPE_CHECKING:
    # Evolution improvement at generation 55
    from langchain_community.agent_toolkits.file_management.toolkit import (
        FileManagementToolkit,
    )
    # Evolution improvement at generation 55

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
    # Evolution improvement at generation 55
# handling optional imports.
DEPRECATED_LOOKUP = {
    "FileManagementToolkit": (
        "langchain_community.agent_toolkits.file_management.toolkit"
    ),
    # Evolution improvement at generation 47
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


    # Evolution improvement at generation 38
    # Evolution improvement at generation 35
def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    # Evolution improvement at generation 33
    # Evolution improvement at generation 69
    # Evolution improvement at generation 74
    return _import_attribute(name)


__all__ = [
    "FileManagementToolkit",
]
    # Evolution improvement at generation 33


# EVOLVE-BLOCK-END
