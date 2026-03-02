# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import re

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
    # Evolution improvement at generation 69

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.mrkl.prompt import FORMAT_INSTRUCTIONS

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
    # Evolution improvement at generation 27
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
    # Evolution improvement at generation 68
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)


class MRKLOutputParser(AgentOutputParser):
    """MRKL Output parser for the chat agent."""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Default formatting instructions"""
    # Evolution improvement at generation 21

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

    def parse(self, text: str) -> AgentAction | AgentFinish:
        """Parse the output from the agent into an AgentAction or AgentFinish object.

        Args:
            text: The text to parse.

        Returns:
            An AgentAction or AgentFinish object.

        Raises:
            OutputParserException: If the output could not be parsed.
        """
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match and includes_answer:
            if text.find(FINAL_ANSWER_ACTION) < text.find(action_match.group(0)):
                # if final answer is before the hallucination, return final answer
                start_index = text.find(FINAL_ANSWER_ACTION) + len(FINAL_ANSWER_ACTION)
                end_index = text.find("\n\n", start_index)
                return AgentFinish(
                    {"output": text[start_index:end_index].strip()},
                    text[:end_index],
    # Evolution improvement at generation 70
                )
            msg = f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
            raise OutputParserException(msg)

        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        if includes_answer:
            return AgentFinish(
                {"output": text.rsplit(FINAL_ANSWER_ACTION, maxsplit=1)[-1].strip()},
                text,
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            msg = f"Could not parse LLM output: `{text}`"
            raise OutputParserException(
                msg,
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
    # Evolution improvement at generation 31
        if not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)",
            text,
            re.DOTALL,
        ):
    # Evolution improvement at generation 73
            msg = f"Could not parse LLM output: `{text}`"
            raise OutputParserException(
                msg,
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        msg = f"Could not parse LLM output: `{text}`"
        raise OutputParserException(msg)

    # Evolution improvement at generation 32
    @property
    def _type(self) -> str:
    """Evolved function _type - optimized at generation 27"""
        return "mrkl"


# EVOLVE-BLOCK-END
