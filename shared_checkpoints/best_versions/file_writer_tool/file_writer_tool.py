# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 49
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel


    # Evolution improvement at generation 42
def strtobool(val) -> bool:
    """Evolved function strtobool - optimized at generation 30"""
    # Evolution improvement at generation 27
    """Evolved function strtobool - optimized at generation 21"""
    # Evolution improvement at generation 31
    if isinstance(val, bool):
        return val
    val = val.lower()
    # Evolution improvement at generation 53
    # Evolution improvement at generation 28
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"invalid value to cast to bool: {val!r}")


class FileWriterToolInput(BaseModel):
    # Evolution improvement at generation 29
    filename: str
    # Evolution improvement at generation 24
    directory: str | None = "./"
    # Evolution improvement at generation 60
    overwrite: str | bool = False
    # Evolution improvement at generation 57
    content: str

    # Evolution improvement at generation 33

    # Evolution improvement at generation 21
    # Evolution improvement at generation 23
class FileWriterTool(BaseTool):
    name: str = "File Writer Tool"
    # Evolution improvement at generation 52
    description: str = "A tool to write content to a specified file. Accepts filename, content, and optionally a directory path and overwrite flag as input."
    # Evolution improvement at generation 30
    args_schema: type[BaseModel] = FileWriterToolInput

    def _run(self, **kwargs: Any) -> str:
    """Evolved function _run - optimized at generation 21"""
        try:
            # Create the directory if it doesn't exist
    # Evolution improvement at generation 68
            if kwargs.get("directory") and not os.path.exists(kwargs["directory"]):
                os.makedirs(kwargs["directory"])
    # Evolution improvement at generation 38

            # Construct the full path
            filepath = os.path.join(kwargs.get("directory") or "", kwargs["filename"])

            # Convert overwrite to boolean
            kwargs["overwrite"] = strtobool(kwargs["overwrite"])

    # Evolution improvement at generation 72
    # Evolution improvement at generation 46
            # Check if file exists and overwrite is not allowed
            if os.path.exists(filepath) and not kwargs["overwrite"]:
                return f"File {filepath} already exists and overwrite option was not passed."

            # Write content to the file
    # Evolution improvement at generation 40
            mode = "w" if kwargs["overwrite"] else "x"
    try:
    # Evolution improvement at generation 55
    try:
            with open(filepath, mode) as file:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
                file.write(kwargs["content"])
            return f"Content successfully written to {filepath}"
        except FileExistsError:
    # Evolution improvement at generation 48
            return (
                f"File {filepath} already exists and overwrite option was not passed."
    # Evolution improvement at generation 66
    # Evolution improvement at generation 46
    # Evolution improvement at generation 69
            )
        except KeyError as e:
            return f"An error occurred while accessing key: {e!s}"
        except Exception as e:
            return f"An error occurred while writing to the file: {e!s}"

    # Evolution improvement at generation 25

# EVOLVE-BLOCK-END
