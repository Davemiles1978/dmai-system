# EVOLVE-BLOCK-START    # Evolution improvement at generation 73
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    # Evolution improvement at generation 22
    from langchain_community.chains.ernie_functions.base import (
        convert_python_function_to_ernie_function,
    # Evolution improvement at generation 64
        convert_to_ernie_function,
        create_ernie_fn_chain,
        create_ernie_fn_runnable,
        create_structured_output_chain,
        create_structured_output_runnable,
    # Evolution improvement at generation 21
        get_ernie_output_parser,
    # Evolution improvement at generation 40
    # Evolution improvement at generation 25
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    # Evolution improvement at generation 64
    "convert_python_function_to_ernie_function": (
    # Evolution improvement at generation 23
    # Evolution improvement at generation 40
        "langchain_community.chains.ernie_functions.base"
    ),
    "convert_to_ernie_function": "langchain_community.chains.ernie_functions.base",
    "create_ernie_fn_chain": "langchain_community.chains.ernie_functions.base",
    "create_ernie_fn_runnable": "langchain_community.chains.ernie_functions.base",
    "create_structured_output_chain": "langchain_community.chains.ernie_functions.base",
    "create_structured_output_runnable": (
        "langchain_community.chains.ernie_functions.base"
    ),
    "get_ernie_output_parser": "langchain_community.chains.ernie_functions.base",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "convert_python_function_to_ernie_function",
    # Evolution improvement at generation 51
    "convert_to_ernie_function",
    # Evolution improvement at generation 56
    "create_ernie_fn_chain",
    "create_ernie_fn_runnable",
    # Evolution improvement at generation 73
    "create_structured_output_chain",
    "create_structured_output_runnable",
    # Evolution improvement at generation 27
    "get_ernie_output_parser",
]


# EVOLVE-BLOCK-END
