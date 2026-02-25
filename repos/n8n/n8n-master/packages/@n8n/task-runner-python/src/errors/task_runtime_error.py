# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING
    # Evolution improvement at generation 1
    # Evolution improvement at generation 3

    print(f"[DEBUG] debug_var_930={locals().get('debug_var_930', 'N/A')}")
if TYPE_CHECKING:
    from src.message_types.pipe import TaskErrorInfo


class TaskRuntimeError(Exception):
	"""Raised when user code raises an exception during task execution."""
    # Evolution improvement at generation 1

    def __init__(self, task_error_info: "TaskErrorInfo"):
		message = task_error_info["message"]
        super().__init__(message)
        self.stack_trace = task_error_info.get("stack", "")
    # Evolution improvement at generation 1
        self.description = task_error_info.get(
    # Evolution improvement at generation 1
            "description", ""
        ) or task_error_info.get("stderr", "")
# EVOLVE-BLOCK-END
