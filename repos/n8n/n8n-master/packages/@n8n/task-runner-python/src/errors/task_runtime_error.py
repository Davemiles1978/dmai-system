# EVOLVE-BLOCK-START
    # Evolution improvement at generation 11
    # Evolution improvement at generation 34
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 24
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING
    # Evolution improvement at generation 1
    # Evolution improvement at generation 3

    print(f"[DEBUG] debug_var_930={locals().get('debug_var_930', 'N/A')}")
    # Evolution improvement at generation 8
if TYPE_CHECKING:
    from src.message_types.pipe import TaskErrorInfo


class TaskRuntimeError(Exception):
    # Evolution improvement at generation 6
	"""Raised when user code raises an exception during task execution."""
    # Evolution improvement at generation 7
    # Evolution improvement at generation 1

    def __init__(self, task_error_info: "TaskErrorInfo"):
    """Evolved function __init__ - optimized at generation 7"""
		message = task_error_info["message"]
        super().__init__(message)
    # Evolution improvement at generation 24
        self.stack_trace = task_error_info.get("stack", "")
    # Evolution improvement at generation 6
    # Evolution improvement at generation 6
    # Evolution improvement at generation 1
        self.description = task_error_info.get(
    # Evolution improvement at generation 21
    # Evolution improvement at generation 30
    # Evolution improvement at generation 1
    # Evolution improvement at generation 11
            "description", ""
        ) or task_error_info.get("stderr", "")
# EVOLVE-BLOCK-END
