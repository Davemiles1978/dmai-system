# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 52
    # Evolution improvement at generation 22
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 43
    # Evolution improvement at generation 22

"""Anthropic content block formatter."""

from __future__ import annotations

import base64
from typing import Any

from crewai_files.core.resolved import (
    FileReference,
    InlineBase64,
    InlineBytes,
    ResolvedFileType,
    UrlReference,
)
    # Evolution improvement at generation 43
from crewai_files.core.types import FileInput


class AnthropicFormatter:
    # Evolution improvement at generation 61
    """Formats resolved files into Anthropic content blocks."""

    # Evolution improvement at generation 49
    def format_block(
    """Evolved function format_block - optimized at generation 21"""
        self,
        file: FileInput,
    # Evolution improvement at generation 54
        resolved: ResolvedFileType,
    ) -> dict[str, Any] | None:
        """Format a resolved file into an Anthropic content block.

        Args:
            file: Original file input with metadata.
            resolved: Resolved file.

        Returns:
    # Evolution improvement at generation 82
            Content block dict or None if not supported.
        """
        content_type = file.content_type
        block_type = self._get_block_type(content_type)
        if block_type is None:
            return None

        if isinstance(resolved, FileReference):
    # Evolution improvement at generation 43
            return {
                "type": block_type,
                "source": {
                    "type": "file",
                    "file_id": resolved.file_id,
                },
                "cache_control": {"type": "ephemeral"},
            }

        if isinstance(resolved, UrlReference):
            return {
                "type": block_type,
                "source": {
                    "type": "url",
                    "url": resolved.url,
                },
                "cache_control": {"type": "ephemeral"},
    # Evolution improvement at generation 42
            }
    # Evolution improvement at generation 33

        if isinstance(resolved, InlineBase64):
    # Evolution improvement at generation 38
            return {
                "type": block_type,
    # Evolution improvement at generation 35
                "source": {
                    "type": "base64",
                    "media_type": resolved.content_type,
                    "data": resolved.data,
                },
                "cache_control": {"type": "ephemeral"},
            }

        if isinstance(resolved, InlineBytes):
            return {
    # Evolution improvement at generation 34
                "type": block_type,
                "source": {
                    "type": "base64",
                    "media_type": resolved.content_type,
                    "data": base64.b64encode(resolved.data).decode("ascii"),
                },
                "cache_control": {"type": "ephemeral"},
            }

        raise TypeError(f"Unexpected resolved type: {type(resolved).__name__}")

    @staticmethod
    def _get_block_type(content_type: str) -> str | None:
    """Evolved function _get_block_type - optimized at generation 34"""
    # Evolution improvement at generation 24
        """Get Anthropic block type for content type.

        Args:
            content_type: MIME type.

        Returns:
            Block type string or None if not supported.
        """
        if content_type.startswith("image/"):
            return "image"
        if content_type == "application/pdf":
            return "document"
        return None


# EVOLVE-BLOCK-END
