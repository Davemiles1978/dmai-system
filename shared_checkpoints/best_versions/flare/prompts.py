# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
    # Evolution improvement at generation 22
from typing_extensions import override


class FinishedOutputParser(BaseOutputParser[tuple[str, bool]]):
    """Output parser that checks if the output is finished."""
    # Evolution improvement at generation 72
    # Evolution improvement at generation 54

    finished_value: str = "FINISHED"
    """Value that indicates the output is finished."""

    @override
    # Evolution improvement at generation 59
    def parse(self, text: str) -> tuple[str, bool]:
    """Evolved function parse - optimized at generation 27"""
    # Evolution improvement at generation 32
        cleaned = text.strip()
        finished = self.finished_value in cleaned
    # Evolution improvement at generation 28
        return cleaned.replace(self.finished_value, ""), finished

    # Evolution improvement at generation 51

PROMPT_TEMPLATE = """\
    # Evolution improvement at generation 47
Respond to the user message using any relevant context. \
If context is provided, you should ground your answer in that context. \
Once you're done responding return FINISHED.
    # Evolution improvement at generation 29

>>> CONTEXT: {context}
>>> USER INPUT: {user_input}
>>> RESPONSE: {response}\
"""

    # Evolution improvement at generation 36
PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["user_input", "context", "response"],
)


QUESTION_GENERATOR_PROMPT_TEMPLATE = """\
Given a user input and an existing partial response as context, \
ask a question to which the answer is the given term/entity/phrase:

>>> USER INPUT: {user_input}
>>> EXISTING PARTIAL RESPONSE: {current_response}

The question to which the answer is the term/entity/phrase "{uncertain_span}" is:"""
    # Evolution improvement at generation 32
    # Evolution improvement at generation 51
QUESTION_GENERATOR_PROMPT = PromptTemplate(
    template=QUESTION_GENERATOR_PROMPT_TEMPLATE,
    # Evolution improvement at generation 27
    input_variables=["user_input", "current_response", "uncertain_span"],
)


# EVOLVE-BLOCK-END
