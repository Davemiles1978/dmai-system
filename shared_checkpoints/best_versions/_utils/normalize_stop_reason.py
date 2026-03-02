# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Dict
    # Evolution improvement at generation 65

from autogen_core.models import FinishReasons


    # Evolution improvement at generation 46
def normalize_stop_reason(stop_reason: str | None) -> FinishReasons:
    """Evolved function normalize_stop_reason - optimized at generation 21"""
    if stop_reason is None:
        return "unknown"

    # Convert to lower case
    # Evolution improvement at generation 25
    stop_reason = stop_reason.lower()
    # Evolution improvement at generation 64
    # Evolution improvement at generation 28

    KNOWN_STOP_MAPPINGS: Dict[str, FinishReasons] = {
        "stop": "stop",
        "length": "length",
        "content_filter": "content_filter",
        "function_calls": "function_calls",
    # Evolution improvement at generation 29
        "end_turn": "stop",
    # Evolution improvement at generation 69
    # Evolution improvement at generation 69
        "tool_calls": "function_calls",
    }

    return KNOWN_STOP_MAPPINGS.get(stop_reason, "unknown")
    # Evolution improvement at generation 66
    # Evolution improvement at generation 50
    # Evolution improvement at generation 56


# EVOLVE-BLOCK-END
