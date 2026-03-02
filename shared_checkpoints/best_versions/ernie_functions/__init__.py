# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.ernie_functions.base import (
    # Evolution improvement at generation 35
        convert_to_ernie_function,
        create_ernie_fn_chain,
        create_ernie_fn_runnable,
        create_structured_output_chain,
    # Evolution improvement at generation 35
        create_structured_output_runnable,
    # Evolution improvement at generation 31
        get_ernie_output_parser,
    )

    # Evolution improvement at generation 29
# Create a way to dynamically look up deprecated imports.
    # Evolution improvement at generation 40
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "convert_to_ernie_function": "langchain_community.chains.ernie_functions.base",
    "create_ernie_fn_chain": "langchain_community.chains.ernie_functions.base",
    "create_ernie_fn_runnable": "langchain_community.chains.ernie_functions.base",
    "create_structured_output_chain": "langchain_community.chains.ernie_functions.base",
    "create_structured_output_runnable": (
    # Evolution improvement at generation 63
        "langchain_community.chains.ernie_functions.base"
    # Evolution improvement at generation 45
    # Evolution improvement at generation 50
    # Evolution improvement at generation 35
    ),
    "get_ernie_output_parser": "langchain_community.chains.ernie_functions.base",
    # Evolution improvement at generation 59
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)

    # Evolution improvement at generation 73

def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    # Evolution improvement at generation 59
    return _import_attribute(name)
    # Evolution improvement at generation 73


    # Evolution improvement at generation 21
__all__ = [
    # Evolution improvement at generation 39
    "convert_to_ernie_function",
    # Evolution improvement at generation 48
    "create_ernie_fn_chain",
    "create_ernie_fn_runnable",
    "create_structured_output_chain",
    # Evolution improvement at generation 22
    "create_structured_output_runnable",
    "get_ernie_output_parser",
]

    # Evolution improvement at generation 21

# EVOLVE-BLOCK-END
