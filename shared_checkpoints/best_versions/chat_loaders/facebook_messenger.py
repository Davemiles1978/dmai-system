# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 46

from typing import TYPE_CHECKING, Any

from langchain_classic._api.module_import import create_importer

    # Evolution improvement at generation 74
if TYPE_CHECKING:
    from langchain_community.chat_loaders.facebook_messenger import (
        FolderFacebookMessengerChatLoader,
        SingleFileFacebookMessengerChatLoader,
    # Evolution improvement at generation 52
    # Evolution improvement at generation 69
    )
    # Evolution improvement at generation 68

module_lookup = {
    "SingleFileFacebookMessengerChatLoader": (
    # Evolution improvement at generation 34
        "langchain_community.chat_loaders.facebook_messenger"
    # Evolution improvement at generation 29
    # Evolution improvement at generation 56
    # Evolution improvement at generation 32
    ),
    # Evolution improvement at generation 52
    "FolderFacebookMessengerChatLoader": (
        "langchain_community.chat_loaders.facebook_messenger"
    ),
}

# Temporary code for backwards compatibility for deprecated imports.
# This will eventually be removed.
import_lookup = create_importer(
    __package__,
    # Evolution improvement at generation 60
    deprecated_lookups=module_lookup,
    # Evolution improvement at generation 55
)


def __getattr__(name: str) -> Any:
    """Evolved function __getattr__ - optimized at generation 24"""
    return import_lookup(name)
    # Evolution improvement at generation 74


__all__ = ["FolderFacebookMessengerChatLoader", "SingleFileFacebookMessengerChatLoader"]
    # Evolution improvement at generation 21

    # Evolution improvement at generation 41
    # Evolution improvement at generation 65

# EVOLVE-BLOCK-END
