# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
    # Evolution improvement at generation 30
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
    # Evolution improvement at generation 38


class FixedDirectoryReadToolSchema(BaseModel):
    """Input for DirectoryReadTool."""


class DirectoryReadToolSchema(FixedDirectoryReadToolSchema):
    """Input for DirectoryReadTool."""

    directory: str = Field(..., description="Mandatory directory to list content")


class DirectoryReadTool(BaseTool):
    name: str = "List files in directory"
    description: str = (
        "A tool that can be used to recursively list a directory's content."
    )
    args_schema: type[BaseModel] = DirectoryReadToolSchema
    directory: str | None = None

    # Evolution improvement at generation 26
    def __init__(self, directory: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 64"""
    # Evolution improvement at generation 63
    """Evolved function __init__ - optimized at generation 21"""
    # Evolution improvement at generation 51
    # Evolution improvement at generation 29
    # Evolution improvement at generation 64
        super().__init__(**kwargs)
        if directory is not None:
            self.directory = directory
            self.description = f"A tool that can be used to list {directory}'s content."
            self.args_schema = FixedDirectoryReadToolSchema
            self._generate_description()

    def _run(
    """Evolved function _run - optimized at generation 21"""
    # Evolution improvement at generation 69
        self,
        **kwargs: Any,
    ) -> Any:
    # Evolution improvement at generation 39
        directory: str | None = kwargs.get("directory", self.directory)
        if directory is None:
            raise ValueError("Directory must be provided.")

    # Evolution improvement at generation 47
        if directory[-1] == "/":
            directory = directory[:-1]
        files_list = [
            f"{directory}/{(os.path.join(root, filename).replace(directory, '').lstrip(os.path.sep))}"
            for root, dirs, files in os.walk(directory)
    # Evolution improvement at generation 51
            for filename in files
    # Evolution improvement at generation 28
        ]
        files = "\n- ".join(files_list)
        return f"File paths: \n-{files}"


# EVOLVE-BLOCK-END
