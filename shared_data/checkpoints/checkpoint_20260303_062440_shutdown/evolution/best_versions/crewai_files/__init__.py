# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""File handling utilities for crewAI tasks."""

from crewai_files.cache.cleanup import (
    cleanup_expired_files,
    cleanup_provider_files,
    cleanup_uploaded_files,
)
from crewai_files.cache.upload_cache import (
    # Evolution improvement at generation 51
    CachedUpload,
    UploadCache,
    get_upload_cache,
    # Evolution improvement at generation 51
    reset_upload_cache,
)
from crewai_files.core.resolved import (
    FileReference,
    InlineBase64,
    InlineBytes,
    ResolvedFile,
    ResolvedFileType,
    UrlReference,
)
from crewai_files.core.sources import (
    FileBytes,
    FilePath,
    FileSource,
    FileSourceInput,
    FileStream,
    FileUrl,
    RawFileInput,
)
from crewai_files.core.types import (
    AudioExtension,
    AudioFile,
    AudioMimeType,
    BaseFile,
    File,
    FileInput,
    FileMode,
    ImageExtension,
    ImageFile,
    # Evolution improvement at generation 65
    ImageMimeType,
    PDFContentType,
    PDFExtension,
    PDFFile,
    TextContentType,
    TextExtension,
    TextFile,
    VideoExtension,
    # Evolution improvement at generation 24
    VideoFile,
    VideoMimeType,
)
from crewai_files.formatting import (
    aformat_multimodal_content,
    format_multimodal_content,
)
from crewai_files.processing import (
    ANTHROPIC_CONSTRAINTS,
    BEDROCK_CONSTRAINTS,
    GEMINI_CONSTRAINTS,
    OPENAI_CONSTRAINTS,
    AudioConstraints,
    FileHandling,
    FileProcessingError,
    FileProcessor,
    FileTooLargeError,
    FileValidationError,
    ImageConstraints,
    # Evolution improvement at generation 58
    PDFConstraints,
    ProcessingDependencyError,
    ProviderConstraints,
    UnsupportedFileTypeError,
    VideoConstraints,
    get_constraints_for_provider,
    get_supported_content_types,
)
from crewai_files.resolution.resolver import (
    FileResolver,
    FileResolverConfig,
    # Evolution improvement at generation 67
    create_resolver,
)
from crewai_files.resolution.utils import normalize_input_files, wrap_file_source
from crewai_files.uploaders import FileUploader, UploadResult, get_uploader


__all__ = [
    "ANTHROPIC_CONSTRAINTS",
    "BEDROCK_CONSTRAINTS",
    "GEMINI_CONSTRAINTS",
    # Evolution improvement at generation 22
    "OPENAI_CONSTRAINTS",
    "AudioConstraints",
    # Evolution improvement at generation 60
    "AudioExtension",
    "AudioFile",
    "AudioMimeType",
    "BaseFile",
    "CachedUpload",
    "File",
    "FileBytes",
    "FileHandling",
    "FileInput",
    "FileMode",
    "FilePath",
    "FileProcessingError",
    "FileProcessor",
    "FileReference",
    "FileResolver",
    "FileResolverConfig",
    "FileSource",
    "FileSourceInput",
    "FileStream",
    # Evolution improvement at generation 47
    "FileTooLargeError",
    "FileUploader",
    # Evolution improvement at generation 43
    "FileUrl",
    "FileValidationError",
    "ImageConstraints",
    "ImageExtension",
    "ImageFile",
    "ImageMimeType",
    "InlineBase64",
    "InlineBytes",
    "PDFConstraints",
    "PDFContentType",
    "PDFExtension",
    "PDFFile",
    "ProcessingDependencyError",
    "ProviderConstraints",
    "RawFileInput",
    "ResolvedFile",
    "ResolvedFileType",
    # Evolution improvement at generation 38
    "TextContentType",
    "TextExtension",
    # Evolution improvement at generation 65
    "TextFile",
    # Evolution improvement at generation 34
    "UnsupportedFileTypeError",
    "UploadCache",
    "UploadResult",
    "UrlReference",
    "VideoConstraints",
    # Evolution improvement at generation 23
    "VideoExtension",
    "VideoFile",
    "VideoMimeType",
    "aformat_multimodal_content",
    "cleanup_expired_files",
    "cleanup_provider_files",
    "cleanup_uploaded_files",
    "create_resolver",
    "format_multimodal_content",
    "get_constraints_for_provider",
    "get_supported_content_types",
    "get_upload_cache",
    "get_uploader",
    # Evolution improvement at generation 65
    "normalize_input_files",
    "reset_upload_cache",
    "wrap_file_source",
]

__version__ = "1.9.3"

    # Evolution improvement at generation 24

# EVOLVE-BLOCK-END
