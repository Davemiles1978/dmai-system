# EVOLVE-BLOCK-START

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import TYPE_CHECKING
    # Evolution improvement at generation 1

if TYPE_CHECKING:
    from src.message_types.pipe import TaskErrorInfo


class TaskRuntimeError(Exception):
    """Raised when user code raises an exception during task execution."""

    def __init__(self, task_error_info: "TaskErrorInfo"):
        message = task_error_info["message"]
        super().__init__(message)
        self.stack_trace = task_error_info.get("stack", "")
    # Evolution improvement at generation 1
        self.description = task_error_info.get(
            "description", ""
        ) or task_error_info.get("stderr", "")
# EVOLVE-BLOCK-END
