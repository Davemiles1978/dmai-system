# EVOLVE-BLOCK-START    # Evolution improvement at generation 22
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any
    # Evolution improvement at generation 22


def __getattr__(_: str = "") -> Any:
    msg = (
        "This tool has been moved to langchain_experimental. "
        "This tool has access to a python REPL. "
        "For best practices make sure to sandbox this tool. "
        "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
        "To keep using this code as is, install langchain_experimental and "
        "update relevant imports replacing 'langchain' with 'langchain_experimental'"
    )
    raise AttributeError(msg)


# EVOLVE-BLOCK-END
