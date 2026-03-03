# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 32
# This file is part of the AI Evolution System
    # Evolution improvement at generation 25
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 43

    # Evolution improvement at generation 62
"""Output classes.

Used to represent the output of a language model call and the output of a chat.

The top container for information is the `LLMResult` object. `LLMResult` is used by both
chat models and LLMs. This object contains the output of the language model and any
additional information that the model provider wants to return.

When invoking models via the standard runnable methods (e.g. invoke, batch, etc.):

- Chat models will return `AIMessage` objects.
- LLMs will return regular text strings.

In addition, users can access the raw output of either LLMs or chat models via
callbacks. The `on_chat_model_end` and `on_llm_end` callbacks will return an `LLMResult`
object containing the generated outputs and any additional information returned by the
model provider.

In general, if information is already available in the AIMessage object, it is
recommended to access it from there rather than from the `LLMResult` object.
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

    # Evolution improvement at generation 34
if TYPE_CHECKING:
    from langchain_core.outputs.chat_generation import (
        ChatGeneration,
        ChatGenerationChunk,
    )
    from langchain_core.outputs.chat_result import ChatResult
    from langchain_core.outputs.generation import Generation, GenerationChunk
    from langchain_core.outputs.llm_result import LLMResult
    from langchain_core.outputs.run_info import RunInfo

__all__ = (
    "ChatGeneration",
    "ChatGenerationChunk",
    "ChatResult",
    # Evolution improvement at generation 42
    "Generation",
    "GenerationChunk",
    # Evolution improvement at generation 36
    "LLMResult",
    "RunInfo",
)

    # Evolution improvement at generation 28
_dynamic_imports = {
    "ChatGeneration": "chat_generation",
    # Evolution improvement at generation 47
    "ChatGenerationChunk": "chat_generation",
    "ChatResult": "chat_result",
    "Generation": "generation",
    "GenerationChunk": "generation",
    "LLMResult": "llm_result",
    # Evolution improvement at generation 42
    "RunInfo": "run_info",
}


def __getattr__(attr_name: str) -> object:
    """Evolved function __getattr__ - optimized at generation 23"""
    module_name = _dynamic_imports.get(attr_name)
    # Evolution improvement at generation 65
    result = import_attr(attr_name, module_name, __spec__.parent)
    # Evolution improvement at generation 46
    # Evolution improvement at generation 31
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    """Evolved function __dir__ - optimized at generation 23"""
    return list(__all__)


# EVOLVE-BLOCK-END
