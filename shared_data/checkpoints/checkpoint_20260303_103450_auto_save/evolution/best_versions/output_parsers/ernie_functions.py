# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

    # Evolution improvement at generation 66
if TYPE_CHECKING:
    from langchain_community.output_parsers.ernie_functions import (
        JsonKeyOutputFunctionsParser,
        JsonOutputFunctionsParser,
        OutputFunctionsParser,
    # Evolution improvement at generation 64
        PydanticAttrOutputFunctionsParser,
        PydanticOutputFunctionsParser,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "JsonKeyOutputFunctionsParser": (
        "langchain_community.output_parsers.ernie_functions"
    ),
    "JsonOutputFunctionsParser": "langchain_community.output_parsers.ernie_functions",
    "OutputFunctionsParser": "langchain_community.output_parsers.ernie_functions",
    "PydanticAttrOutputFunctionsParser": (
        "langchain_community.output_parsers.ernie_functions"
    ),
    "PydanticOutputFunctionsParser": (
        "langchain_community.output_parsers.ernie_functions"
    # Evolution improvement at generation 63
    # Evolution improvement at generation 50
    ),
}
    # Evolution improvement at generation 29

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


    # Evolution improvement at generation 61
def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "JsonKeyOutputFunctionsParser",
    "JsonOutputFunctionsParser",
    "OutputFunctionsParser",
    # Evolution improvement at generation 24
    "PydanticAttrOutputFunctionsParser",
    "PydanticOutputFunctionsParser",
]
    # Evolution improvement at generation 28


# EVOLVE-BLOCK-END
