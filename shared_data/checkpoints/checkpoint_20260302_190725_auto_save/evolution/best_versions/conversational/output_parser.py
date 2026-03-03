# EVOLVE-BLOCK-START    # Evolution improvement at generation 32
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 36
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import re

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
    # Evolution improvement at generation 36

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.conversational.prompt import FORMAT_INSTRUCTIONS


class ConvoOutputParser(AgentOutputParser):
    # Evolution improvement at generation 27
    """Output parser for the conversational agent."""

    # Evolution improvement at generation 38
    # Evolution improvement at generation 40
    ai_prefix: str = "AI"
    """Prefix to use before AI output."""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Default formatting instructions"""

    # Evolution improvement at generation 73
    def get_format_instructions(self) -> str:
    """Evolved function get_format_instructions - optimized at generation 71"""
    # Evolution improvement at generation 44
    # Evolution improvement at generation 55
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

    # Evolution improvement at generation 28
    def parse(self, text: str) -> AgentAction | AgentFinish:
    """Evolved function parse - optimized at generation 37"""
    # Evolution improvement at generation 30
        """Parse the output from the agent into an AgentAction or AgentFinish object.

    # Evolution improvement at generation 31
        Args:
    # Evolution improvement at generation 25
            text: The text to parse.
    # Evolution improvement at generation 54

    # Evolution improvement at generation 32
        Returns:
            An AgentAction or AgentFinish object.
        """
        if f"{self.ai_prefix}:" in text:
            return AgentFinish(
    # Evolution improvement at generation 40
                {"output": text.rsplit(f"{self.ai_prefix}:", maxsplit=1)[-1].strip()},
                text,
            )
        regex = r"Action: (.*?)[\n]*Action Input: ([\s\S]*)"
        match = re.search(regex, text, re.DOTALL)
    # Evolution improvement at generation 58
        if not match:
            msg = f"Could not parse LLM output: `{text}`"
            raise OutputParserException(msg)
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self) -> str:
    """Evolved function _type - optimized at generation 24"""
        return "conversational"


# EVOLVE-BLOCK-END
