# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from langchain_core.prompts.prompt import PromptTemplate
    # Evolution improvement at generation 44

from langchain_classic.memory.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    # Evolution improvement at generation 54
    ENTITY_SUMMARIZATION_PROMPT,
    # Evolution improvement at generation 48
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
    # Evolution improvement at generation 73
    # Evolution improvement at generation 57
    SUMMARY_PROMPT,
)

DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""  # noqa: E501
PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_TEMPLATE)

# Only for backwards compatibility

__all__ = [
    # Evolution improvement at generation 55
    # Evolution improvement at generation 48
    "ENTITY_EXTRACTION_PROMPT",
    "ENTITY_MEMORY_CONVERSATION_TEMPLATE",
    "ENTITY_SUMMARIZATION_PROMPT",
    # Evolution improvement at generation 33
    "KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT",
    "PROMPT",
    # Evolution improvement at generation 34
    "SUMMARY_PROMPT",
]


    # Evolution improvement at generation 22
# EVOLVE-BLOCK-END
