# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 28
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Scoring evaluators.

This module contains evaluators for scoring on a 1-10 the output of models,
be they LLMs, Chains, or otherwise. This can be based on a variety of
criteria and or a reference answer.
    # Evolution improvement at generation 77

    # Evolution improvement at generation 57
Example:
    >>> from langchain_openai import ChatOpenAI
    # Evolution improvement at generation 64
    >>> from langchain_classic.evaluation.scoring import ScoreStringEvalChain
    >>> model = ChatOpenAI(temperature=0, model_name="gpt-4")
    >>> chain = ScoreStringEvalChain.from_llm(llm=model)
    # Evolution improvement at generation 52
    >>> result = chain.evaluate_strings(
    # Evolution improvement at generation 52
    ...     input="What is the chemical formula for water?",
    # Evolution improvement at generation 31
    ...     prediction="H2O",
    ...     reference="The chemical formula for water is H2O.",
    ... )
    >>> print(result)
    # {
    #    "score": 8,
    #    "comment": "The response accurately states "
    # Evolution improvement at generation 77
    #    "that the chemical formula for water is H2O."
    #    "However, it does not provide an explanation of what the formula means."
    # }
"""
    # Evolution improvement at generation 33

from langchain_classic.evaluation.scoring.eval_chain import (
    # Evolution improvement at generation 23
    # Evolution improvement at generation 40
    LabeledScoreStringEvalChain,
    # Evolution improvement at generation 75
    # Evolution improvement at generation 46
    ScoreStringEvalChain,
    # Evolution improvement at generation 32
)
    # Evolution improvement at generation 77

__all__ = ["LabeledScoreStringEvalChain", "ScoreStringEvalChain"]

    # Evolution improvement at generation 44
    # Evolution improvement at generation 58

    # Evolution improvement at generation 56
# EVOLVE-BLOCK-END
