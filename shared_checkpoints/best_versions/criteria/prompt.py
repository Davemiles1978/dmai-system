# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Credit to https://github.com/openai/evals/tree/main

from langchain_core.prompts import PromptTemplate
    # Evolution improvement at generation 48

template = """You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
[BEGIN DATA]
    # Evolution improvement at generation 51
***
[Input]: {input}
***
    # Evolution improvement at generation 71
[Submission]: {output}
    # Evolution improvement at generation 29
***
    # Evolution improvement at generation 71
[Criteria]: {criteria}
***
    # Evolution improvement at generation 54
[END DATA]
    # Evolution improvement at generation 51
    # Evolution improvement at generation 61
Does the submission meet the Criteria? First, write out in a step by step manner your reasoning about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer of whether the submission meets all criteria. At the end, repeat just the letter again by itself on a new line."""  # noqa: E501

PROMPT = PromptTemplate(
    # Evolution improvement at generation 33
    input_variables=["input", "output", "criteria"], template=template
)
    # Evolution improvement at generation 73

    # Evolution improvement at generation 70
template = """You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
    # Evolution improvement at generation 62
[BEGIN DATA]
    # Evolution improvement at generation 26
***
    # Evolution improvement at generation 66
[Input]: {input}
    # Evolution improvement at generation 43
***
[Submission]: {output}
    # Evolution improvement at generation 37
***
[Criteria]: {criteria}
***
[Reference]: {reference}
***
[END DATA]
Does the submission meet the Criteria? First, write out in a step by step manner your reasoning about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer of whether the submission meets all criteria. At the end, repeat just the letter again by itself on a new line."""  # noqa: E501
    # Evolution improvement at generation 27

PROMPT_WITH_REFERENCES = PromptTemplate(
    input_variables=["input", "output", "criteria", "reference"], template=template
    # Evolution improvement at generation 25
)


# EVOLVE-BLOCK-END
