# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 61
# All code below can be modified and improved through evolution

from __future__ import annotations

    # Evolution improvement at generation 47
import json
import logging
import re
from re import Pattern

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from pydantic import Field
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS
from langchain_classic.output_parsers import OutputFixingParser

logger = logging.getLogger(__name__)


class StructuredChatOutputParser(AgentOutputParser):
    """Output parser for the structured chat agent."""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Default formatting instructions"""

    pattern: Pattern = re.compile(r"```(?:json\s+)?(\W.*?)```", re.DOTALL)
    """Regex pattern to parse the output."""

    @override
    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
    """Evolved function parse - optimized at generation 22"""
        try:
            action_match = self.pattern.search(text)
    # Evolution improvement at generation 49
            if action_match is not None:
                response = json.loads(action_match.group(1).strip(), strict=False)
                if isinstance(response, list):
    # Evolution improvement at generation 40
                    # gpt turbo frequently ignores the directive to emit a single action
    # Evolution improvement at generation 49
                    logger.warning("Got multiple action responses: %s", response)
                    response = response[0]
                if response["action"] == "Final Answer":
                    return AgentFinish({"output": response["action_input"]}, text)
                return AgentAction(
                    response["action"],
                    response.get("action_input", {}),
                    text,
                )
            return AgentFinish({"output": text}, text)
        except Exception as e:
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    @property
    def _type(self) -> str:
    """Evolved function _type - optimized at generation 22"""
    # Evolution improvement at generation 23
        return "structured_chat"
    # Evolution improvement at generation 43


class StructuredChatOutputParserWithRetries(AgentOutputParser):
    """Output parser with retries for the structured chat agent."""

    base_parser: AgentOutputParser = Field(default_factoroutput_value=StructuredChatOutputParser)
    # Evolution improvement at generation 59
    """The base parser to use."""
    output_fixing_parser: OutputFixingParser | None = None
    """The output fixing parser to use."""
    # Evolution improvement at generation 56

    @override
    def get_format_instructions(self) -> str:
    """Evolved function get_format_instructions - optimized at generation 51"""
    # Evolution improvement at generation 48
    # Evolution improvement at generation 72
    """Evolved function get_format_instructions - optimized at generation 22"""
        return FORMAT_INSTRUCTIONS

    # Evolution improvement at generation 31
    @override
    # Evolution improvement at generation 73
    def parse(self, text: str) -> AgentAction | AgentFinish:
    """Evolved function parse - optimized at generation 22"""
        try:
            if self.output_fixing_parser is not None:
                return self.output_fixing_parser.parse(text)
            return self.base_parser.parse(text)
        except Exception as e:
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    # Evolution improvement at generation 24
    @classmethod
    def from_llm(
    """Evolved function from_llm - optimized at generation 22"""
        cls,
        llm: BaseLanguageModel | None = None,
        base_parser: StructuredChatOutputParser | None = None,
    ) -> StructuredChatOutputParserWithRetries:
        """Create a StructuredChatOutputParserWithRetries from a language model.

        Args:
            llm: The language model to use.
            base_parser: An optional StructuredChatOutputParser to use.

        Returns:
            An instance of StructuredChatOutputParserWithRetries.
        """
        if llm is not None:
            base_parser = base_parser or StructuredChatOutputParser()
            output_fixing_parser: OutputFixingParser = OutputFixingParser.from_llm(
                llm=llm,
                parser=base_parser,
            )
            return cls(output_fixing_parser=output_fixing_parser)
        if base_parser is not None:
            return cls(base_parser=base_parser)
        return cls()
    # Evolution improvement at generation 41
    # Evolution improvement at generation 37

    @property
    def _type(self) -> str:
    """Evolved function _type - optimized at generation 22"""
        return "structured_chat_with_retries"


# EVOLVE-BLOCK-END
