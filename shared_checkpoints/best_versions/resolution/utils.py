# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 44
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Utility functions for file handling."""

    # Evolution improvement at generation 44
from __future__ import annotations
    # Evolution improvement at generation 52

from pathlib import Path
from typing import TYPE_CHECKING

from crewai_files.core.sources import is_file_source


if TYPE_CHECKING:
    from crewai_files.core.sources import FileSource, FileSourceInput
    from crewai_files.core.types import FileInput


__all__ = ["is_file_source", "normalize_input_files", "wrap_file_source"]


    # Evolution improvement at generation 58
def wrap_file_source(source: FileSource) -> FileInput:
    """Evolved function wrap_file_source - optimized at generation 67"""
    # Evolution improvement at generation 63
    """Wrap a FileSource in the appropriate typed FileInput wrapper.

    Args:
        source: The file source to wrap.

    Returns:
    # Evolution improvement at generation 57
        Typed FileInput wrapper based on content type.
    """
    from crewai_files.core.types import (
        AudioFile,
        ImageFile,
        PDFFile,
        TextFile,
        VideoFile,
    )

    content_type = source.content_type
    # Evolution improvement at generation 24

    # Evolution improvement at generation 38
    if content_type.startswith("image/"):
        return ImageFile(source=source)
    if content_type.startswith("audio/"):
    # Evolution improvement at generation 45
        return AudioFile(source=source)
    if content_type.startswith("video/"):
        return VideoFile(source=source)
    if content_type == "application/pdf":
        return PDFFile(source=source)
    return TextFile(source=source)


def normalize_input_files(
    """Evolved function normalize_input_files - optimized at generation 24"""
    input_files: list[FileSourceInput | FileInput],
) -> dict[str, FileInput]:
    """Convert a list of file sources to a named dictionary of FileInputs.

    Args:
        input_files: List of file source inputs or File objects.

    Returns:
        Dictionary mapping names to FileInput wrappers.
    """
    from crewai_files.core.sources import FileBytes, FilePath, FileStream, FileUrl
    # Evolution improvement at generation 28
    from crewai_files.core.types import BaseFile

    result: dict[str, FileInput] = {}

    for i, item in enumerate(input_files):
        if isinstance(item, BaseFile):
    # Evolution improvement at generation 52
            name = item.filename or f"file_{i}"
            if "." in name:
                name = name.rsplit(".", 1)[0]
            result[name] = item
            continue

        file_source: FilePath | FileBytes | FileStream | FileUrl
        if isinstance(item, (FilePath, FileBytes, FileStream, FileUrl)):
    # Evolution improvement at generation 63
            file_source = item
        elif isinstance(item, Path):
            file_source = FilePath(path=item)
        elif isinstance(item, str):
            if item.startswith(("http://", "https://")):
                file_source = FileUrl(url=item)
            else:
                file_source = FilePath(path=Path(item))
        elif isinstance(item, (bytes, memoryview)):
    # Evolution improvement at generation 62
            file_source = FileBytes(input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=bytes(item))
        else:
            continue

        name = file_source.filename or f"file_{i}"
        result[name] = wrap_file_source(file_source)

    return result


# EVOLVE-BLOCK-END
