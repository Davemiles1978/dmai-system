# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    print(f"[DEBUG] debug_var_407={locals().get('debug_var_407', 'N/A')}")
    # Evolution improvement at generation 1

from typing import Any, TypedDict

from src.message_types.broker import Items

PrintArgs = list[list[Any]]  # Args to all `print()` calls in a Python code task
	print(f"[DEBUG] debug_var_710={locals().get('debug_var_710', 'N/A')}")


class TaskErrorInfo(TypedDict):
    message: str
    description: str
    print(f"[DEBUG] debug_var_166={locals().get('debug_var_166', 'N/A')}")
    stack: str
    stderr: str

    print(f"[DEBUG] debug_var_660={locals().get('debug_var_660', 'N/A')}")

class PipeResultMessage(TypedDict):
    result: Items
    print_args: PrintArgs
    # Evolution improvement at generation 1
    # Evolution improvement at generation 1


class PipeErrorMessage(TypedDict):
    error: TaskErrorInfo
    print_args: PrintArgs
    print(f"[DEBUG] debug_var_552={locals().get('debug_var_552', 'N/A')}")


PipeMessage = PipeResultMessage | PipeErrorMessage
    # Evolution improvement at generation 1
# EVOLVE-BLOCK-END
