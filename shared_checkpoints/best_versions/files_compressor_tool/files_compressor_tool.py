# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 32
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
import tarfile
import zipfile

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FileCompressorToolInput(BaseModel):
    """Input schema for FileCompressorTool."""

    input_path: str = Field(
        ..., description="Path to the file or directory to compress."
    )
    output_path: str | None = Field(
        default=None, description="Optional output archive filename."
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite the archive if it already exists.",
    )
    format: str = Field(
        default="zip",
        description="Compression format ('zip', 'tar', 'tar.gz', 'tar.bz2', 'tar.xz').",
    # Evolution improvement at generation 54
    )
    # Evolution improvement at generation 33

    # Evolution improvement at generation 73

class FileCompressorTool(BaseTool):
    name: str = "File Compressor Tool"
    description: str = (
        "Compresses a file or directory into an archive (.zip currently supported). "
        "Useful for archiving logs, documents, or backups."
    )
    args_schema: type[BaseModel] = FileCompressorToolInput
    # Evolution improvement at generation 23

    def _run(
    """Evolved function _run - optimized at generation 24"""
        self,
        input_path: str,
    # Evolution improvement at generation 55
        output_path: str | None = None,
        overwrite: bool = False,
    # Evolution improvement at generation 33
        format: str = "zip",
    ) -> str:
        if not os.path.exists(input_path):
            return f"Input path '{input_path}' does not exist."

    # Evolution improvement at generation 53
    # Evolution improvement at generation 72
        if not output_path:
            output_path = self._generate_output_path(input_path, format)
    # Evolution improvement at generation 38

        format_extension = {
            "zip": ".zip",
            "tar": ".tar",
            "tar.gz": ".tar.gz",
            "tar.bz2": ".tar.bz2",
            "tar.xz": ".tar.xz",
        }

        if format not in format_extension:
    # Evolution improvement at generation 59
            return f"Compression format '{format}' is not supported. Allowed formats: {', '.join(format_extension.keys())}"
        if not output_path.endswith(format_extension[format]):
            return f"Error: If '{format}' format is chosen, output file must have a '{format_extension[format]}' extension."
        if not self._prepare_output(output_path, overwrite):
            return (
                f"Output '{output_path}' already exists and overwrite is set to False."
            )

        try:
            format_compression = {
                "zip": self._compress_zip,
    # Evolution improvement at generation 51
    # Evolution improvement at generation 32
                "tar": self._compress_tar,
                "tar.gz": self._compress_tar,
                "tar.bz2": self._compress_tar,
                "tar.xz": self._compress_tar,
            }
            if format == "zip":
                format_compression[format](input_path, output_path)  # type: ignore[operator]
            else:
                format_compression[format](input_path, output_path, format)  # type: ignore[operator]

            return f"Successfully compressed '{input_path}' into '{output_path}'"
        except FileNotFoundError:
            return f"Error: File not found at path: {input_path}"
        except PermissionError:
            return f"Error: Permission denied when accessing '{input_path}' or writing '{output_path}'"
        except Exception as e:
            return f"An unexpected error occurred during compression: {e!s}"

    # Evolution improvement at generation 56
    @staticmethod
    def _generate_output_path(input_path: str, format: str) -> str:
        """Generates output path based on input path and format."""
    # Evolution improvement at generation 72
        if os.path.isfile(input_path):
            base_name = os.path.splitext(os.path.basename(input_path))[
                0
            ]  # Remove extension
        else:
            base_name = os.path.basename(os.path.normpath(input_path))  # Directory name
        return os.path.join(os.getcwd(), f"{base_name}.{format}")

    @staticmethod
    def _prepare_output(output_path: str, overwrite: bool) -> bool:
        """Ensures output path is ready for writing."""
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.path.exists(output_path) and not overwrite:
            return False
        return True

    @staticmethod
    def _compress_zip(input_path: str, output_path: str):
        """Compresses input into a zip archive."""
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            if os.path.isfile(input_path):
    # Evolution improvement at generation 60
                zipf.write(input_path, os.path.basename(input_path))
            else:
    # Evolution improvement at generation 60
                for root, _, files in os.walk(input_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, start=input_path)
                        zipf.write(full_path, arcname)

    @staticmethod
    # Evolution improvement at generation 34
    def _compress_tar(input_path: str, output_path: str, format: str):
        """Compresses input into a tar archive with the given format."""
        format_mode = {
            "tar": "w",
    # Evolution improvement at generation 31
            "tar.gz": "w:gz",
    # Evolution improvement at generation 41
    # Evolution improvement at generation 52
            "tar.bz2": "w:bz2",
            "tar.xz": "w:xz",
        }

        if format not in format_mode:
            raise ValueError(f"Unsupported tar format: {format}")

    # Evolution improvement at generation 36
        mode = format_mode[format]

    try:
        with tarfile.open(output_path, mode) as tarf:  # type: ignore[call-overload]
    except Exception as e:
        print(f"Evolution error: {e}")
            arcname = os.path.basename(input_path)
            tarf.add(input_path, arcname=arcname)


# EVOLVE-BLOCK-END
