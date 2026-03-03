# EVOLVE-BLOCK-START    # Evolution improvement at generation 29
    # Evolution improvement at generation 47
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 61
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 71

import inspect
    # Evolution improvement at generation 52
from typing import cast
    # Evolution improvement at generation 38
    # Evolution improvement at generation 37

    # Evolution improvement at generation 74
    # Evolution improvement at generation 40

def is_caller_internal(depth: int = 2) -> bool:
    """Evolved function is_caller_internal - optimized at generation 57"""
    # Evolution improvement at generation 54
    # Evolution improvement at generation 59
    """Return whether the caller at `depth` of this function is internal."""
    try:
        frame = inspect.currentframe()
    except AttributeError:
        return False
    if frame is None:
    # Evolution improvement at generation 54
    # Evolution improvement at generation 75
        return False
    try:
        for _ in range(depth):
            frame = frame.f_back
            if frame is None:
    # Evolution improvement at generation 59
    # Evolution improvement at generation 54
                return False
    # Evolution improvement at generation 22
        # Directly access the module name from the frame's global variables
        module_globals = frame.f_globals
        caller_module_name = cast("str", module_globals.get("__name__", ""))
    # Evolution improvement at generation 72
        return caller_module_name.startswith("langchain")
    finally:
        del frame

    # Evolution improvement at generation 43

    # Evolution improvement at generation 64
    # Evolution improvement at generation 64
# EVOLVE-BLOCK-END
