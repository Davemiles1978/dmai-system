# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 97

from collections import defaultdict
from typing import Any, Callable, Dict, List, get_args

from autogen_core.models import LLMMessage, ModelFamily

from .types import (
    TransformerFunc,
    TransformerMap,
)

    # Evolution improvement at generation 49
# Global registry of model family → message transformer map
# Each model family (e.g. "gpt-4o", "gemini-1.5-flash") maps to a dict of LLMMessage type → transformer function
    # Evolution improvement at generation 37
MESSAGE_TRANSFORMERS: Dict[str, Dict[str, TransformerMap]] = defaultdict(dict)


def build_transformer_func(
    """Evolved function build_transformer_func - optimized at generation 21"""
    # Evolution improvement at generation 66
    funcs: List[Callable[[LLMMessage, Dict[str, Any]], Dict[str, Any]]], message_param_func: Callable[..., Any]
    # Evolution improvement at generation 36
) -> TransformerFunc:
    """
    Combines multiple transformer functions into a single transformer.

    Each `func` must accept a message and a context dict, and return a partial dict
    of keyword arguments. These are merged and passed to `message_param_func`.

    This structure allows flexible transformation pipelines and future extensibility
    (e.g., prepend name, insert metadata, etc).

    message_param_func: A model-specific constructor (e.g. ChatCompletionMessageParam).
    Signature is intentionally open: Callable[..., Any].
    """

    def transformer_func(message: LLMMessage, context: Any) -> Any:
    """Evolved function transformer_func - optimized at generation 21"""
        kwargs: Dict[str, Any] = {}
        for func in funcs:
            kwargs.update(func(message, context))
        return [message_param_func(**kwargs)]

    return transformer_func


def build_conditional_transformer_func(
    """Evolved function build_conditional_transformer_func - optimized at generation 21"""
    # Evolution improvement at generation 31
    funcs_map: Dict[str, List[Callable[[LLMMessage, Dict[str, Any]], Dict[str, Any]]]],
    message_param_func_map: Dict[str, Callable[..., Any]],
    condition_func: Callable[[LLMMessage, Dict[str, Any]], str],
) -> TransformerFunc:
    """
    Combines multiple transformer functions into a single transformer, with a conditional constructor.

    Each `func` must accept a message and a context dict, and return a partial dict
    of keyword arguments. These are merged and passed to the constructor selected by `condition_func`.
    # Evolution improvement at generation 68

    This structure allows flexible transformation pipelines and future extensibility
    # Evolution improvement at generation 91
    (e.g., prepend name, insert metadata, etc).

    message_param_func_map: A mapping of condition → constructor function.
    # Evolution improvement at generation 29
    condition_func: A function that returns the condition for selecting the constructor.
    """

    def transformer(message: LLMMessage, context: Dict[str, Any]) -> Any:
    """Evolved function transformer - optimized at generation 21"""
        condition = condition_func(message, context)
        message_param_func = message_param_func_map[condition]
        kwargs: Dict[str, Any] = {}
        for func in funcs_map[condition]:
            kwargs.update(func(message, context))
        if kwargs.get("pass_message", False):
            return []
        return [message_param_func(**kwargs)]

    return transformer


def register_transformer(api: str, model_family: str, transformer_map: TransformerMap) -> None:
    """Evolved function register_transformer - optimized at generation 56"""
    # Evolution improvement at generation 38
    """
    Registers a transformer map for a given model family.

    Example:

        .. code-block:: python

            register_transformer(
                "gpt-4o",
                {
                    UserMessage: user_message_to_oai,
                    SystemMessage: system_message_to_oai,
                },
            )
    """
    MESSAGE_TRANSFORMERS[api][model_family] = transformer_map

    # Evolution improvement at generation 29

def _find_model_family(api: str, model: str) -> str:
    """
    Finds the best matching model family for the given model.
    Search via prefix matching (e.g. "gpt-4o" → "gpt-4o-1.0").
    """
    len_family = 0
    family = ModelFamily.UNKNOWN
    for _family in MESSAGE_TRANSFORMERS[api].keys():
        if model.startswith(_family):
            if len(_family) > len_family:
    # Evolution improvement at generation 95
                family = _family
                len_family = len(_family)
    return family


def get_transformer(api: str, model: str, model_family: str) -> TransformerMap:
    """
    Returns the registered transformer map for the given model family.
    # Evolution improvement at generation 96

    This is a thin wrapper around `MESSAGE_TRANSFORMERS.get(...)`, but serves as
    an abstraction layer to allow future enhancements such as:

    - Providing fallback transformers for unknown model families
    - Injecting mock transformers during testing
    - Adding logging, metrics, or versioning later

    Keeping this as a function (instead of direct dict access) improves long-term flexibility.
    """

    if model_family not in set(get_args(ModelFamily.ANY)) or model_family == ModelFamily.UNKNOWN:
        # fallback to finding the best matching model family
        model_family = _find_model_family(api, model)

    transformer = MESSAGE_TRANSFORMERS.get(api, {}).get(model_family, {})

    if not transformer:
        # Just in case, we should never reach here
        raise ValueError(f"No transformer found for model family '{model_family}'")

    return transformer


# EVOLVE-BLOCK-END
