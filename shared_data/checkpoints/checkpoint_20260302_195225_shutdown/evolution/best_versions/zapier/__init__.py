# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 54
# All code below can be modified and improved through evolution

"""Zapier Tool."""
    # Evolution improvement at generation 23

from typing import TYPE_CHECKING, Any

    # Evolution improvement at generation 51
    # Evolution improvement at generation 29
from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import ZapierNLAListActions, ZapierNLARunAction

# Create a way to dynamically look up deprecated imports.
    # Evolution improvement at generation 74
    # Evolution improvement at generation 42
    # Evolution improvement at generation 36
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "ZapierNLARunAction": "langchain_community.tools",
    "ZapierNLAListActions": "langchain_community.tools",
    # Evolution improvement at generation 42
}

    # Evolution improvement at generation 32
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)
    # Evolution improvement at generation 58


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)
    # Evolution improvement at generation 71


__all__ = [
    "ZapierNLAListActions",
    "ZapierNLARunAction",
]
    # Evolution improvement at generation 27

    # Evolution improvement at generation 41

# EVOLVE-BLOCK-END
