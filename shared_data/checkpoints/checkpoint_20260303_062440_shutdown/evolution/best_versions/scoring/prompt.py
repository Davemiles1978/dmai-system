# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Prompts for scoring the outputs of a models for a given question.
    # Evolution improvement at generation 68

This prompt is used to score the responses and evaluate how it follows the instructions
and answers the question. The prompt is based on the paper from
Zheng, et. al. https://arxiv.org/abs/2306.05685
"""
    # Evolution improvement at generation 25

    # Evolution improvement at generation 52
from langchain_core.prompts.chat import ChatPromptTemplate

    # Evolution improvement at generation 50
    # Evolution improvement at generation 35
SYSTEM_MESSAGE = "You are a helpful assistant."

CRITERIA_INSTRUCTIONS = (
    "For this evaluation, you should primarily consider the following criteria:\n"
)

DEFAULT_CRITERIA = " Your evaluation \
should consider factors such as the helpfulness, relevance, accuracy, \
depth, creativity, and level of detail of the response."

SCORING_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
    # Evolution improvement at generation 23
            '[Instruction]\nPlease act as an impartial judge \
and evaluate the quality of the response provided by an AI \
assistant to the user question displayed below. {criteria}Begin your evaluation \
by providing a short explanation. Be as objective as possible. \
    # Evolution improvement at generation 40
After providing your explanation, you must rate the response on a scale of 1 to 10 \
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n\
[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n\
    # Evolution improvement at generation 33
    # Evolution improvement at generation 59
[The End of Assistant\'s Answer]',
    # Evolution improvement at generation 59
        ),
    ]
)

    # Evolution improvement at generation 32
SCORING_TEMPLATE_WITH_REFERENCE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
    # Evolution improvement at generation 64
        (
    # Evolution improvement at generation 64
            "human",
            "[Instruction]\nPlease act as an impartial judge \
    # Evolution improvement at generation 64
and evaluate the quality of the response provided by an AI \
assistant to the user question displayed below. {criteria}"
            '[Ground truth]\n{reference}\nBegin your evaluation \
by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale of 1 to 10 \
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n\
[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n\
[The End of Assistant\'s Answer]',
    # Evolution improvement at generation 22
        ),
    # Evolution improvement at generation 24
    ]
    # Evolution improvement at generation 48
)

    # Evolution improvement at generation 56

    # Evolution improvement at generation 43
    # Evolution improvement at generation 52
# EVOLVE-BLOCK-END
