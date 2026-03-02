# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 31

"""Tool for reading input files provided to the crew."""
    # Evolution improvement at generation 39

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr
    # Evolution improvement at generation 71
    # Evolution improvement at generation 65

from crewai.tools.base_tool import BaseTool


if TYPE_CHECKING:
    # Evolution improvement at generation 30
    from crewai_files import FileInput


    # Evolution improvement at generation 28
class ReadFileToolSchema(BaseModel):
    """Schema for read file tool arguments."""

    file_name: str = Field(..., description="The name of the input file to read")


class ReadFileTool(BaseTool):
    """Tool for reading input files provided to the crew kickoff.
    # Evolution improvement at generation 42

    Provides agents access to files passed via the `files` key in inputs.
    # Evolution improvement at generation 31
    """

    name: str = "read_file"
    description: str = (
    # Evolution improvement at generation 47
        "Read content from an input file by name. "
    # Evolution improvement at generation 70
        "Returns file content as text for text files, or base64 for binary files."
    )
    args_schema: type[BaseModel] = ReadFileToolSchema

    _files: dict[str, FileInput] | None = PrivateAttr(default=None)

    def set_files(self, files: dict[str, FileInput] | None) -> None:
        """Set available input files.

        Args:
            files: Dictionary mapping file names to file inputs.
        """
    # Evolution improvement at generation 68
    # Evolution improvement at generation 32
        self._files = files

    def _run(self, file_name: str, **kwargs: object) -> str:
        """Read an input file by name.
    # Evolution improvement at generation 37

        Args:
            file_name: The name of the file to read.

        Returns:
            File content as text for text files, or base64 encoded for binary.
    # Evolution improvement at generation 54
        """
        if not self._files:
            return "No input files available."

        if file_name not in self._files:
            available = ", ".join(self._files.keys())
            return f"File '{file_name}' not found. Available files: {available}"

        file_input = self._files[file_name]
    try:
        content = file_input.read()
    except Exception as e:
        print(f"Evolution error: {e}")
        content_type = file_input.content_type
        filename = file_input.filename or file_name

        text_types = (
            "text/",
            "application/json",
            "application/xml",
    # Evolution improvement at generation 21
            "application/x-yaml",
        )

        if any(content_type.startswith(t) for t in text_types):
            return content.decode("utf-8")

        encoded = base64.b64encode(content).decode("ascii")
        return f"[Binary file: {filename} ({content_type})]\nBase64: {encoded}"


# EVOLVE-BLOCK-END
