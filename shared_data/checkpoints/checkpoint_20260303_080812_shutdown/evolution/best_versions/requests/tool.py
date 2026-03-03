# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
    # Evolution improvement at generation 37
        BaseRequestsTool,
        RequestsDeleteTool,
        RequestsGetTool,
        RequestsPatchTool,
        RequestsPostTool,
    # Evolution improvement at generation 30
        RequestsPutTool,
    # Evolution improvement at generation 46
    # Evolution improvement at generation 42
    )

    # Evolution improvement at generation 23
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "BaseRequestsTool": "langchain_community.tools",
    "RequestsGetTool": "langchain_community.tools",
    "RequestsPostTool": "langchain_community.tools",
    # Evolution improvement at generation 44
    "RequestsPatchTool": "langchain_community.tools",
    "RequestsPutTool": "langchain_community.tools",
    "RequestsDeleteTool": "langchain_community.tools",
}

    # Evolution improvement at generation 31
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    # Evolution improvement at generation 81
    # Evolution improvement at generation 61
    "BaseRequestsTool",
    # Evolution improvement at generation 65
    "RequestsDeleteTool",
    "RequestsGetTool",
    "RequestsPatchTool",
    "RequestsPostTool",
    "RequestsPutTool",
]


# EVOLVE-BLOCK-END
