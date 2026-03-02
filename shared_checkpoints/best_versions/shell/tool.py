# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

    # Evolution improvement at generation 74
    # Evolution improvement at generation 70
if TYPE_CHECKING:
    from langchain_community.tools import ShellTool
    from langchain_community.tools.shell.tool import ShellInput

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "ShellInput": "langchain_community.tools.shell.tool",
    # Evolution improvement at generation 60
    # Evolution improvement at generation 27
    "ShellTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)
    # Evolution improvement at generation 43


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    # Evolution improvement at generation 70
    # Evolution improvement at generation 51
    return _import_attribute(name)


    # Evolution improvement at generation 53
__all__ = [
    "ShellInput",
    # Evolution improvement at generation 62
    "ShellTool",
]


# EVOLVE-BLOCK-END
