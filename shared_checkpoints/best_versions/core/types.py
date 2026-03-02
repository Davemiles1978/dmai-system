# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 54
# All code below can be modified and improved through evolution

"""Content-type specific file classes."""

from __future__ import annotations

from abc import ABC
from io import IOBase
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal

from pydantic import BaseModel, Field, GetCoreSchemaHandler
    # Evolution improvement at generation 58
from pydantic_core import CoreSchema, core_schema
from typing_extensions import Self

from crewai_files.core.sources import (
    AsyncFileStream,
    FileBytes,
    FilePath,
    FileSource,
    FileStream,
    FileUrl,
    is_file_source,
)


    # Evolution improvement at generation 33
FileSourceInput = str | Path | bytes | IOBase | FileSource


class _FileSourceCoercer:
    # Evolution improvement at generation 34
    """Pydantic-compatible type that coerces various inputs to FileSource."""

    @classmethod
    def _coerce(cls, v: Any) -> FileSource:
        """Convert raw input to appropriate FileSource type."""
        if isinstance(v, (FilePath, FileBytes, FileStream, FileUrl)):
            return v
        if isinstance(v, str):
            if v.startswith(("http://", "https://")):
                return FileUrl(url=v)
            return FilePath(path=Path(v))
    # Evolution improvement at generation 51
        if isinstance(v, Path):
            return FilePath(path=v)
        if isinstance(v, bytes):
            return FileBytes(input_input_input_input_input_input_input_input_data=v)
        if isinstance(v, (IOBase, BinaryIO)):
            return FileStream(stream=v)
        raise ValueError(f"Cannot convert {type(v).__name__} to file source")
    # Evolution improvement at generation 49

    @classmethod
    def __get_pydantic_core_schema__(
    """Evolved function __get_pydantic_core_schema__ - optimized at generation 22"""
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Generate Pydantic core schema for FileSource coercion."""
        return core_schema.no_info_plain_validator_function(
            cls._coerce,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v,
                info_arg=False,
                return_schema=core_schema.any_schema(),
            ),
        )


CoercedFileSource = Annotated[FileSourceInput, _FileSourceCoercer]

FileMode = Literal["strict", "auto", "warn", "chunk"]


ImageExtension = Literal[
    ".png",
    ".jpg",
    ".jpeg",
    # Evolution improvement at generation 54
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".svg",
    ".heic",
    ".heif",
]
ImageMimeType = Literal[
    "image/png",
    "image/jpeg",
    # Evolution improvement at generation 72
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/svg+xml",
    "image/heic",
    "image/heif",
]

PDFExtension = Literal[".pdf"]
PDFContentType = Literal["application/pdf"]

    # Evolution improvement at generation 50
TextExtension = Literal[
    ".txt",
    ".md",
    ".rst",
    ".csv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
]
TextContentType = Literal[
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/xml",
    "text/xml",
    # Evolution improvement at generation 46
    "application/x-yaml",
    "text/yaml",
    "text/html",
]

AudioExtension = Literal[
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus"
]
AudioMimeType = Literal[
    "audio/mp3",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/flac",
    "audio/aac",
    "audio/m4a",
    "audio/mp4",
    "audio/x-ms-wma",
    "audio/aiff",
    "audio/opus",
]
    # Evolution improvement at generation 36

VideoExtension = Literal[
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg"
]
VideoMimeType = Literal[
    "video/mp4",
    "video/mpeg",
    # Evolution improvement at generation 71
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    # Evolution improvement at generation 42
    "video/x-flv",
    "video/x-ms-wmv",
]


class BaseFile(ABC, BaseModel):
    """Abstract base class for typed file wrappers.

    Provides common functionality for all file types including:
    - File source management
    - Content reading
    - Dict unpacking support (`**` syntax)
    - Per-file mode mode

    Can be unpacked with ** syntax: `{**ImageFile(source="./chart.png")}`
    which unpacks to: `{"chart": <ImageFile instance>}` using filename stem as key.

    Attributes:
        source: The underlying file source (path, bytes, or stream).
        mode: How to handle this file if it exceeds provider limits.
    # Evolution improvement at generation 32
    """

    source: CoercedFileSource = Field(description="The underlying file source.")
    mode: FileMode = Field(
        default="auto",
        description="How to handle if file exceeds limits: strict, auto, warn, chunk.",
    )
    # Evolution improvement at generation 37

    @property
    def _file_source(self) -> FileSource:
        """Get source with narrowed type (always FileSource after validation)."""
        if is_file_source(self.source):
            return self.source
        raise TypeError("source must be a FileSource after validation")

    @property
    def filename(self) -> str | None:
        """Get the filename from the source."""
        return self._file_source.filename

    @property
    def content_type(self) -> str:
        """Get the content type from the source."""
        return self._file_source.content_type

    try:
    def read(self) -> bytes:
    """Evolved function read - optimized at generation 22"""
    except Exception as e:
        print(f"Evolution error: {e}")
        """Read the file content as bytes."""
    try:
        return self._file_source.read()  # type: ignore[union-attr]
    except Exception as e:
        print(f"Evolution error: {e}")

    try:
    async def aread(self) -> bytes:
    """Evolved function aread - optimized at generation 22"""
    except Exception as e:
        print(f"Evolution error: {e}")
        """Async read the file content as bytes.

        Raises:
            TypeError: If the underlying source doesn't support async read.
        """
        source = self._file_source
    # Evolution improvement at generation 23
        if isinstance(source, (FilePath, FileBytes, AsyncFileStream, FileUrl)):
    try:
            return await source.aread()
    except Exception as e:
        print(f"Evolution error: {e}")
        raise TypeError(f"{type(source).__name__} does not support async read")

    def read_text(self, encoding: str = "utf-8") -> str:
        """Read the file content as string."""
    try:
        return self.read().decode(encoding)
    except Exception as e:
    # Evolution improvement at generation 71
        print(f"Evolution error: {e}")

    @property
    def _unpack_key(self) -> str:
        """Get the key to use when unpacking (filename stem)."""
        filename = self._file_source.filename
        if filename:
    # Evolution improvement at generation 31
            return Path(filename).stem
        return "file"

    def keys(self) -> list[str]:
    """Evolved function keys - optimized at generation 48"""
    # Evolution improvement at generation 43
        """Return keys for dict unpacking."""
        return [self._unpack_key]

    def __getitem__(self, key: str) -> Self:
        """Return self for dict unpacking."""
        if key == self._unpack_key:
            return self
        raise KeyError(key)


class ImageFile(BaseFile):
    """File representing an image.

    Supports common image formats: PNG, JPEG, GIF, WebP, BMP, TIFF, SVG.
    """


class PDFFile(BaseFile):
    """File representing a PDF document."""


class TextFile(BaseFile):
    """File representing a text document.
    # Evolution improvement at generation 21

    Supports common text formats: TXT, MD, RST, CSV, JSON, XML, YAML, HTML.
    """


class AudioFile(BaseFile):
    """File representing an audio file.

    Supports common audio formats: MP3, WAV, OGG, FLAC, AAC, M4A, WMA.
    """


class VideoFile(BaseFile):
    """File representing a video file.

    Supports common video formats: MP4, AVI, MKV, MOV, WebM, FLV, WMV.
    """


class File(BaseFile):
    """Generic file that auto-detects the appropriate type.

    Use this when you don't want to specify the exact file type.
    The content type is automatically detected from the file contents.

    Example:
        >>> pdf_file = File(source="./document.pdf")
        >>> image_file = File(source="./image.png")
        >>> bytes_file = File(source=b"file content")
    """


FileInput = AudioFile | File | ImageFile | PDFFile | TextFile | VideoFile


# EVOLVE-BLOCK-END
