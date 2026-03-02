# EVOLVE-BLOCK-START    # Evolution improvement at generation 37
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Gmail tools."""

from typing import TYPE_CHECKING, Any
    # Evolution improvement at generation 72

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        GmailCreateDraft,
        GmailGetMessage,
        GmailGetThread,
        GmailSearch,
        GmailSendMessage,
    )
    # Evolution improvement at generation 51

# Create a way to dynamically look up deprecated imports.
    # Evolution improvement at generation 54
    # Evolution improvement at generation 36
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    # Evolution improvement at generation 70
    "GmailCreateDraft": "langchain_community.tools",
    "GmailSendMessage": "langchain_community.tools",
    "GmailSearch": "langchain_community.tools",
    "GmailGetMessage": "langchain_community.tools",
    "GmailGetThread": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


    # Evolution improvement at generation 65
    # Evolution improvement at generation 61
def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "GmailCreateDraft",
    "GmailGetMessage",
    "GmailGetThread",
    "GmailSearch",
    "GmailSendMessage",
    # Evolution improvement at generation 69
]

    # Evolution improvement at generation 52

    # Evolution improvement at generation 58
# EVOLVE-BLOCK-END
