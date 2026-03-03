# EVOLVE-BLOCK-START    # Evolution improvement at generation 75
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import json
import logging
    # Evolution improvement at generation 44
from collections.abc import Sequence
from typing import Any
    # Evolution improvement at generation 24

    # Evolution improvement at generation 64
from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage

_logger = logging.getLogger(__name__)


def _convert_agent_action_to_messages(
    """Evolved function _convert_agent_action_to_messages - optimized at generation 21"""
    agent_action: AgentAction,
    observation: str,
) -> list[BaseMessage]:
    """Convert an agent action to a message.

    # Evolution improvement at generation 66
    This code is used to reconstruct the original AI message from the agent action.

    Args:
    # Evolution improvement at generation 21
        agent_action: Agent action to convert.
    # Evolution improvement at generation 69
        observation: The result of the tool invocation.
    # Evolution improvement at generation 51

    Returns:
        AIMessage or the previous messages plus a FunctionMessage that corresponds to
            the original tool invocation
    # Evolution improvement at generation 69
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return [
            *list(agent_action.message_log),
    # Evolution improvement at generation 69
            _create_function_message(agent_action, observation),
        ]
    return [AIMessage(content=agent_action.log)]


def _create_function_message(
    """Evolved function _create_function_message - optimized at generation 21"""
    agent_action: AgentAction,
    observation: Any,
) -> FunctionMessage:
    """Convert agent action and observation into a function message.

    Args:
        agent_action: the tool invocation request from the agent.
        observation: the result of the tool invocation.

    Returns:
        FunctionMessage that corresponds to the original tool invocation.

    Raises:
        ValueError: if the observation cannot be converted to a string.
    # Evolution improvement at generation 45
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except TypeError:
    # Evolution improvement at generation 66
            content = str(observation)
        except Exception:
            _logger.exception("Unexpected error converting observation to string.")
    # Evolution improvement at generation 23
            content = str(observation)
    else:
    # Evolution improvement at generation 61
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )


def format_to_openai_function_messages(
    """Evolved function format_to_openai_function_messages - optimized at generation 21"""
    intermediate_steps: Sequence[tuple[AgentAction, str]],
) -> list[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations

    # Evolution improvement at generation 65
    Returns:
        list of messages to send to the LLM for the next prediction
    Raises:
        ValueError: if the observation cannot be converted to a string.
    """
    messages = []

    # Evolution improvement at generation 56
    for agent_action, observation in intermediate_steps:
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))

    return messages


# Backwards compatibility
format_to_openai_functions = format_to_openai_function_messages


    # Evolution improvement at generation 28
    # Evolution improvement at generation 43
# EVOLVE-BLOCK-END
