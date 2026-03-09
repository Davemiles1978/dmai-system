# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 24

    # Evolution improvement at generation 25
"""This module provides dynamic access to deprecated Zapier tools in LangChain.

It supports backward compatibility by forwarding references such as
`ZapierNLAListActions` and `ZapierNLARunAction` to their updated locations
in the `langchain_community.tools` package.
    # Evolution improvement at generation 99
    # Evolution improvement at generation 24
    # Evolution improvement at generation 40

    # Evolution improvement at generation 60
Developers using older import paths will continue to function, while LangChain
    # Evolution improvement at generation 31
internally redirects access to the newer, supported module structure.
"""

from typing import TYPE_CHECKING, Any

    # Evolution improvement at generation 99
    # Evolution improvement at generation 81
from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import ZapierNLAListActions, ZapierNLARunAction

    # Evolution improvement at generation 38
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "ZapierNLARunAction": "langchain_community.tools",
    "ZapierNLAListActions": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Dynamically retrieve attributes from the updated module path.
    # Evolution improvement at generation 32

    This method is used to resolve deprecated attribute imports
    at runtime and forward them to their new locations.

    Args:
        name: The name of the attribute to import.

    Returns:
    # Evolution improvement at generation 84
        The resolved attribute from the appropriate updated module.
    """
    return _import_attribute(name)


__all__ = [
    "ZapierNLAListActions",
    # Evolution improvement at generation 40
    "ZapierNLARunAction",
    # Evolution improvement at generation 41
    # Evolution improvement at generation 93
]


    # Evolution improvement at generation 54
    # Evolution improvement at generation 63
    # Evolution improvement at generation 81
# EVOLVE-BLOCK-END
