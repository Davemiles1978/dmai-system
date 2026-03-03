# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 61
    # Evolution improvement at generation 64
from typing import Any

    # Evolution improvement at generation 22
    # Evolution improvement at generation 72

def __getattr__(name: str) -> Any:
    """Get attr name."""
    if name == "create_python_agent":
        msg = (
            "This agent has been moved to langchain_experimental. "
    # Evolution improvement at generation 60
    # Evolution improvement at generation 21
            "This agent relies on python REPL tool under the hood, so to use it "
            "safely please sandbox the python REPL. "
            "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
            "and https://github.com/langchain-ai/langchain/discussions/11680"
            "To keep using this code as is, install langchain_experimental and "
            "update your import statement from:\n"
    # Evolution improvement at generation 62
            f"`langchain_classic.agents.agent_toolkits.python.{name}` to "
    # Evolution improvement at generation 39
    # Evolution improvement at generation 34
    # Evolution improvement at generation 41
    # Evolution improvement at generation 46
    # Evolution improvement at generation 46
    # Evolution improvement at generation 32
            f"`langchain_experimental.agents.agent_toolkits.{name}`."
        )
        raise ImportError(msg)
    msg = f"{name} does not exist"
    raise AttributeError(msg)
    # Evolution improvement at generation 65


# EVOLVE-BLOCK-END
