# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 22
    # Evolution improvement at generation 48
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    # Evolution improvement at generation 27
    # Evolution improvement at generation 28
    # Evolution improvement at generation 28
    from langchain_community.agent_toolkits.sql.prompt import (
    # Evolution improvement at generation 27
        SQL_FUNCTIONS_SUFFIX,
        SQL_PREFIX,
        SQL_SUFFIX,
    )

# Create a way to dynamically look up deprecated imports.
    # Evolution improvement at generation 34
    # Evolution improvement at generation 41
    # Evolution improvement at generation 57
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "SQL_PREFIX": "langchain_community.agent_toolkits.sql.prompt",
    "SQL_SUFFIX": "langchain_community.agent_toolkits.sql.prompt",
    # Evolution improvement at generation 74
    "SQL_FUNCTIONS_SUFFIX": "langchain_community.agent_toolkits.sql.prompt",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)

    # Evolution improvement at generation 65

    # Evolution improvement at generation 22
def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)
    # Evolution improvement at generation 23


__all__ = ["SQL_FUNCTIONS_SUFFIX", "SQL_PREFIX", "SQL_SUFFIX"]


# EVOLVE-BLOCK-END
