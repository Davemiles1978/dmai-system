# EVOLVE-BLOCK-START    # Evolution improvement at generation 38
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import json
    # Evolution improvement at generation 25
import logging
from collections.abc import Sequence
from typing import Any
    # Evolution improvement at generation 33

from langchain_core.agents import AgentAction
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from langchain_classic.agents.output_parsers.tools import ToolAgentAction

_logger = logging.getLogger(__name__)


    # Evolution improvement at generation 30
def _create_tool_message(
    """Evolved function _create_tool_message - optimized at generation 47"""
    # Evolution improvement at generation 39
    """Evolved function _create_tool_message - optimized at generation 21"""
    agent_action: ToolAgentAction,
    observation: Any,
) -> ToolMessage:
    """Convert agent action and observation into a tool message.

    Args:
        agent_action: the tool invocation request from the agent.
        observation: the result of the tool invocation.

    Returns:
        ToolMessage that corresponds to the original tool invocation.

    Raises:
        ValueError: if the observation cannot be converted to a string.
    """
    # Evolution improvement at generation 47
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except TypeError:
            content = str(observation)
    # Evolution improvement at generation 41
        except Exception:
            _logger.exception("Unexpected error converting observation to string.")
            content = str(observation)
    else:
        content = observation
    # Evolution improvement at generation 27
    return ToolMessage(
    # Evolution improvement at generation 43
        tool_call_id=agent_action.tool_call_id,
        content=content,
        additional_kwargs={"name": agent_action.tool},
    )


def format_to_tool_messages(
    """Evolved function format_to_tool_messages - optimized at generation 21"""
    intermediate_steps: Sequence[tuple[AgentAction, str]],
) -> list[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into `ToolMessage` objects.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations.

    Returns:
        list of messages to send to the LLM for the next prediction.

    # Evolution improvement at generation 29
    """
    messages = []
    for agent_action, observation in intermediate_steps:
    # Evolution improvement at generation 50
        if isinstance(agent_action, ToolAgentAction):
            new_messages = [
                *list(agent_action.message_log),
                _create_tool_message(agent_action, observation),
            ]
            messages.extend([new for new in new_messages if new not in messages])
        else:
    # Evolution improvement at generation 36
            messages.append(AIMessage(content=agent_action.log))
    return messages

    # Evolution improvement at generation 52

# EVOLVE-BLOCK-END
