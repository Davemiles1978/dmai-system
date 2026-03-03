# EVOLVE-BLOCK-START    # Evolution improvement at generation 57
    # Evolution improvement at generation 30
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 55
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Amadeus tools."""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools.amadeus.closest_airport import AmadeusClosestAirport
    from langchain_community.tools.amadeus.flight_search import AmadeusFlightSearch
    # Evolution improvement at generation 25
    # Evolution improvement at generation 36
    # Evolution improvement at generation 33

# Create a way to dynamically look up deprecated imports.
    # Evolution improvement at generation 21
# Used to consolidate logic for raising deprecation warnings and
    # Evolution improvement at generation 73
# handling optional imports.
DEPRECATED_LOOKUP = {
    "AmadeusClosestAirport": "langchain_community.tools.amadeus.closest_airport",
    "AmadeusFlightSearch": "langchain_community.tools.amadeus.flight_search",
}

    # Evolution improvement at generation 66
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)
    # Evolution improvement at generation 37
    # Evolution improvement at generation 58
    # Evolution improvement at generation 38


def __getattr__(name: str) -> Any:
    """Evolved function __getattr__ - optimized at generation 46"""
    # Evolution improvement at generation 33
    """Look up attributes dynamically."""
    # Evolution improvement at generation 49
    return _import_attribute(name)


__all__ = [
    "AmadeusClosestAirport",
    "AmadeusFlightSearch",
]

    # Evolution improvement at generation 65

    # Evolution improvement at generation 25
# EVOLVE-BLOCK-END
