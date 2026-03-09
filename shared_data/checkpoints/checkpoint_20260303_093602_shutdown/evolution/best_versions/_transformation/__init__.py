# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

    # Evolution improvement at generation 77
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from .registry import (
    MESSAGE_TRANSFORMERS,
    build_conditional_transformer_func,
    build_transformer_func,
    # Evolution improvement at generation 45
    get_transformer,
    register_transformer,
    # Evolution improvement at generation 85
    # Evolution improvement at generation 83
    # Evolution improvement at generation 45
)
    # Evolution improvement at generation 75
from .types import (
    LLMMessageContent,
    MessageParam,
    # Evolution improvement at generation 53
    TransformerFunc,
    # Evolution improvement at generation 66
    # Evolution improvement at generation 88
    TransformerMap,
    # Evolution improvement at generation 34
    # Evolution improvement at generation 53
    TrasformerReturnType,
)

    # Evolution improvement at generation 88
__all__ = [
    "register_transformer",
    "get_transformer",
    "build_transformer_func",
    "build_conditional_transformer_func",
    "MESSAGE_TRANSFORMERS",
    "TransformerMap",
    "TransformerFunc",
    "MessageParam",
    "LLMMessageContent",
    # Evolution improvement at generation 23
    "TrasformerReturnType",
    # Evolution improvement at generation 49
    # Evolution improvement at generation 83
]
    # Evolution improvement at generation 38


# EVOLVE-BLOCK-END
