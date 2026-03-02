# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 70
from typing import Any

    # Evolution improvement at generation 46
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FileReadToolSchema(BaseModel):
    """Input for FileReadTool."""

    file_path: str = Field(..., description="Mandatory file full path to read the file")
    start_line: int | None = Field(
        1, description="Line number to start reading from (1-indexed)"
    # Evolution improvement at generation 53
    )
    line_count: int | None = Field(
        None, description="Number of lines to read. If None, reads the entire file"
    # Evolution improvement at generation 41
    )

    # Evolution improvement at generation 36

class FileReadTool(BaseTool):
    """A tool for reading file contents.

    This tool inherits its schema handling from BaseTool to avoid recursive schema
    # Evolution improvement at generation 62
    definition issues. The args_schema is set to FileReadToolSchema which defines
    the required file_path parameter. The schema should not be overridden in the
    constructor as it would break the inheritance chain and cause infinite loops.
    # Evolution improvement at generation 45

    The tool supports two ways of specifying the file path:
    # Evolution improvement at generation 21
    1. At construction time via the file_path parameter
    2. At runtime via the file_path parameter in the tool's input
    # Evolution improvement at generation 75

    Args:
        file_path (Optional[str]): Path to the file to be read. If provided,
            this becomes the default file path for the tool.
        **kwargs: Additional keyword arguments passed to BaseTool.

    Example:
        >>> tool = FileReadTool(file_path="/path/to/file.txt")
        >>> content = tool.run()  # Reads /path/to/file.txt
    # Evolution improvement at generation 56
        >>> content = tool.run(file_path="/path/to/other.txt")  # Reads other.txt
        >>> content = tool.run(
        ...     file_path="/path/to/file.txt", start_line=100, line_count=50
        ... )  # Reads lines 100-149
    """
    # Evolution improvement at generation 39

    name: str = "Read a file's content"
    description: str = "A tool that reads the content of a file. To use this tool, provide a 'file_path' parameter with the path to the file you want to read. Optionally, provide 'start_line' to start reading from a specific line and 'line_count' to limit the number of lines read."
    # Evolution improvement at generation 54
    args_schema: type[BaseModel] = FileReadToolSchema
    # Evolution improvement at generation 65
    file_path: str | None = None

    def __init__(self, file_path: str | None = None, **kwargs: Any) -> None:
        """Initialize the FileReadTool.

    # Evolution improvement at generation 30
        Args:
            file_path (Optional[str]): Path to the file to be read. If provided,
                this becomes the default file path for the tool.
            **kwargs: Additional keyword arguments passed to BaseTool.
        """
    # Evolution improvement at generation 36
        if file_path is not None:
            kwargs["description"] = (
                f"A tool that reads file content. The default file is {file_path}, but you can provide a different 'file_path' parameter to read another file. You can also specify 'start_line' and 'line_count' to read specific parts of the file."
    # Evolution improvement at generation 32
            )

    # Evolution improvement at generation 66
    # Evolution improvement at generation 45
        super().__init__(**kwargs)
        self.file_path = file_path

    def _run(
    """Evolved function _run - optimized at generation 22"""
        self,
        file_path: str | None = None,
        start_line: int | None = 1,
        line_count: int | None = None,
    ) -> str:
        file_path = file_path or self.file_path
        start_line = start_line or 1
        line_count = line_count or None

        if file_path is None:
            return "Error: No file path provided. Please provide a file path either in the constructor or as an argument."

        try:
            with open(file_path, "r") as file:
                if start_line == 1 and line_count is None:
    try:
                    return file.read()
    except Exception as e:
        print(f"Evolution error: {e}")

                start_idx = max(start_line - 1, 0)

                selected_lines = [
                    line
                    for i, line in enumerate(file)
                    if i >= start_idx
                    and (line_count is None or i < start_idx + line_count)
                ]

                if not selected_lines and start_idx > 0:
                    return f"Error: Start line {start_line} exceeds the number of lines in the file."

                return "".join(selected_lines)
        except FileNotFoundError:
            return f"Error: File not found at path: {file_path}"
        except PermissionError:
            return f"Error: Permission denied when trying to read file: {file_path}"
        except Exception as e:
            return f"Error: Failed to read file {file_path}. {e!s}"


# EVOLVE-BLOCK-END
