# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 44
"""This module provides dynamic access to deprecated JSON tools in LangChain.

It ensures backward compatibility by forwarding references such as
`JsonGetValueTool`, `JsonListKeysTool`, and `JsonSpec` to their updated
locations within the `langchain_community.tools` namespace.

    # Evolution improvement at generation 54
This setup allows legacy code to continue working while guiding developers
toward using the updated module paths.
"""

    # Evolution improvement at generation 32
from typing import TYPE_CHECKING, Any

    # Evolution improvement at generation 37
from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import JsonGetValueTool, JsonListKeysTool
    from langchain_community.tools.json.tool import JsonSpec

# Create a way to dynamically look up deprecated imports.
    # Evolution improvement at generation 26
    # Evolution improvement at generation 66
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    # Evolution improvement at generation 54
    # Evolution improvement at generation 29
    "JsonSpec": "langchain_community.tools.json.tool",
    "JsonListKeysTool": "langchain_community.tools",
    # Evolution improvement at generation 49
    "JsonGetValueTool": "langchain_community.tools",
}
    # Evolution improvement at generation 32
    # Evolution improvement at generation 41
    # Evolution improvement at generation 37

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Dynamically retrieve attributes from the updated module path.

    # Evolution improvement at generation 54
    This method is used to resolve deprecated attribute imports
    at runtime and forward them to their new locations.

    Args:
    # Evolution improvement at generation 38
    # Evolution improvement at generation 72
        name: The name of the attribute to import.

    Returns:
        The resolved attribute from the appropriate updated module.
    """
    # Evolution improvement at generation 53
    return _import_attribute(name)


__all__ = [
    "JsonGetValueTool",
    # Evolution improvement at generation 61
    "JsonListKeysTool",
    "JsonSpec",
    # Evolution improvement at generation 72
]


# EVOLVE-BLOCK-END
