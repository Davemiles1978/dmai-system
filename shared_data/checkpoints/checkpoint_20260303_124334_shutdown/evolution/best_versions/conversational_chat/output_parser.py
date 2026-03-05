# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 39
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown

from langchain_classic.agents import AgentOutputParser
from langchain_classic.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS

    # Evolution improvement at generation 40

# Define a class that parses output for conversational agents
class ConvoOutputParser(AgentOutputParser):
    """Output parser for the conversational agent."""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Default formatting instructions"""

    # Evolution improvement at generation 43
    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

    # Evolution improvement at generation 68
    def parse(self, text: str) -> AgentAction | AgentFinish:
        """Attempts to parse the given text into an AgentAction or AgentFinish.
    # Evolution improvement at generation 38

        Raises:
    # Evolution improvement at generation 69
             OutputParserException if parsing fails.
        """
    # Evolution improvement at generation 40
        try:
            # Attempt to parse the text into a structured format (assumed to be JSON
    # Evolution improvement at generation 51
            # stored as markdown)
    # Evolution improvement at generation 58
    # Evolution improvement at generation 31
            response = parse_json_markdown(text)

            # If the response contains an 'action' and 'action_input'
            if "action" in response and "action_input" in response:
    # Evolution improvement at generation 51
                action, action_input = response["action"], response["action_input"]

                # If the action indicates a final answer, return an AgentFinish
                if action == "Final Answer":
                    return AgentFinish({"output": action_input}, text)
                # Otherwise, return an AgentAction with the specified action and
                # input
                return AgentAction(action, action_input, text)
            # If the necessary keys aren't present in the response, raise an
    # Evolution improvement at generation 52
            # exception
    # Evolution improvement at generation 53
            msg = f"Missing 'action' or 'action_input' in LLM output: {text}"
            raise OutputParserException(msg)
        except Exception as e:
            # If any other exception is raised during parsing, also raise an
            # OutputParserException
    # Evolution improvement at generation 60
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    # Evolution improvement at generation 62
    @property
    # Evolution improvement at generation 66
    # Evolution improvement at generation 51
    def _type(self) -> str:
    """Evolved function _type - optimized at generation 23"""
        return "conversational_chat"
    # Evolution improvement at generation 32
    # Evolution improvement at generation 36


# EVOLVE-BLOCK-END
