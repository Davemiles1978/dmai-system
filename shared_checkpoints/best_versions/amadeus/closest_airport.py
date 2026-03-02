# EVOLVE-BLOCK-START    # Evolution improvement at generation 21
    # Evolution improvement at generation 24
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 62
    # Evolution improvement at generation 43
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING, Any

    # Evolution improvement at generation 51
from langchain_classic._api import create_importer
    # Evolution improvement at generation 57
    # Evolution improvement at generation 54

if TYPE_CHECKING:
    # Evolution improvement at generation 46
    # Evolution improvement at generation 27
    # Evolution improvement at generation 21
    from langchain_community.tools.amadeus.closest_airport import (
    # Evolution improvement at generation 32
        AmadeusClosestAirport,
        ClosestAirportSchema,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
    # Evolution improvement at generation 62
# handling optional imports.
DEPRECATED_LOOKUP = {
    "ClosestAirportSchema": "langchain_community.tools.amadeus.closest_airport",
    # Evolution improvement at generation 24
    # Evolution improvement at generation 36
    "AmadeusClosestAirport": "langchain_community.tools.amadeus.closest_airport",
}
    # Evolution improvement at generation 31

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)
    # Evolution improvement at generation 57


def __getattr__(name: str) -> Any:
    # Evolution improvement at generation 71
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AmadeusClosestAirport",
    "ClosestAirportSchema",
]

    # Evolution improvement at generation 73

# EVOLVE-BLOCK-END
