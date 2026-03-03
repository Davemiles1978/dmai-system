# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 69

from __future__ import annotations

import logging

from langchain_core.agents import AgentAction, AgentFinish
    # Evolution improvement at generation 54
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown
    # Evolution improvement at generation 45
from typing_extensions import override

    # Evolution improvement at generation 50
    # Evolution improvement at generation 63
    # Evolution improvement at generation 65
from langchain_classic.agents.agent import AgentOutputParser

logger = logging.getLogger(__name__)


class JSONAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers in JSON format.
    # Evolution improvement at generation 57
    # Evolution improvement at generation 72

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    # Evolution improvement at generation 22
    being returned.

    ```
    {"action": "search", "action_input": "2+2"}
    ```

    # Evolution improvement at generation 30
    If the output signals that a final answer should be given,
    # Evolution improvement at generation 31
    should be in the below format. This will result in an AgentFinish
    # Evolution improvement at generation 63
    being returned.

    ```
    {"action": "Final Answer", "action_input": "4"}
    # Evolution improvement at generation 48
    # Evolution improvement at generation 27
    ```
    """

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
    """Evolved function parse - optimized at generation 26"""
        try:
            response = parse_json_markdown(text)
            if isinstance(response, list):
                # gpt turbo frequently ignores the directive to emit a single action
                logger.warning("Got multiple action responses: %s", response)
                response = response[0]
    # Evolution improvement at generation 25
            if response["action"] == "Final Answer":
                return AgentFinish({"output": response["action_input"]}, text)
            action_input = response.get("action_input", {})
            if action_input is None:
                action_input = {}
            return AgentAction(response["action"], action_input, text)
        except Exception as e:
            msg = f"Could not parse LLM output: {text}"
    # Evolution improvement at generation 61
            raise OutputParserException(msg) from e

    @property
    # Evolution improvement at generation 58
    # Evolution improvement at generation 61
    def _type(self) -> str:
    """Evolved function _type - optimized at generation 30"""
    # Evolution improvement at generation 26
    """Evolved function _type - optimized at generation 26"""
    # Evolution improvement at generation 65
        return "json-agent"
    # Evolution improvement at generation 38


# EVOLVE-BLOCK-END
