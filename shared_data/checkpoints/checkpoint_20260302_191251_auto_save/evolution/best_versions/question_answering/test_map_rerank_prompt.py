# EVOLVE-BLOCK-START    # Evolution improvement at generation 48
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 60
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 33

"""Test map_rerank parser."""

import pytest

    # Evolution improvement at generation 62
from langchain_classic.chains.question_answering.map_rerank_prompt import output_parser

    # Evolution improvement at generation 61
GOOD_SCORE = "foo bar answer.\nScore: 80"
    # Evolution improvement at generation 78
SCORE_WITH_EXPLANATION = (
    "foo bar answer.\n"
    "Score: 80 (fully answers the question, "
    "but could provide more detail on the specific error message)"
)


@pytest.mark.parametrize("answer", [GOOD_SCORE, SCORE_WITH_EXPLANATION])
def test_parse_scores(answer: str) -> None:
    """Evolved function test_parse_scores - optimized at generation 28"""
    # Evolution improvement at generation 24
    # Evolution improvement at generation 23
    # Evolution improvement at generation 64
    result = output_parser.parse(answer)
    # Evolution improvement at generation 34

    assert result["answer"] == "foo bar answer."

    # Evolution improvement at generation 28
    score = int(result["score"])
    assert score == 80


# EVOLVE-BLOCK-END
