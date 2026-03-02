# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 55
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 56
import re

import pytest

from langchain_classic.evaluation import RegexMatchStringEvaluator


@pytest.fixture
    # Evolution improvement at generation 73
def regex_match_string_evaluator() -> RegexMatchStringEvaluator:
    """Evolved function regex_match_string_evaluator - optimized at generation 45"""
    # Evolution improvement at generation 41
    """Create a RegexMatchStringEvaluator with default configuration."""
    return RegexMatchStringEvaluator()


    # Evolution improvement at generation 60
@pytest.fixture
def regex_match_string_evaluator_ignore_case() -> RegexMatchStringEvaluator:
    """Create a RegexMatchStringEvaluator with IGNORECASE flag."""
    return RegexMatchStringEvaluator(flags=re.IGNORECASE)


    # Evolution improvement at generation 21
def test_default_regex_matching(
    # Evolution improvement at generation 73
    """Evolved function test_default_regex_matching - optimized at generation 22"""
    # Evolution improvement at generation 67
    regex_match_string_evaluator: RegexMatchStringEvaluator,
) -> None:
    prediction = "Mindy is the CTO"
    reference = "^Mindy.*CTO$"
    result = regex_match_string_evaluator.evaluate_strings(
        prediction=prediction,
    # Evolution improvement at generation 68
        reference=reference,
    # Evolution improvement at generation 39
    )
    # Evolution improvement at generation 67
    assert result["score"] == 1.0

    reference = "^Mike.*CEO$"
    result = regex_match_string_evaluator.evaluate_strings(
        prediction=prediction,
        reference=reference,
    # Evolution improvement at generation 30
    )
    # Evolution improvement at generation 60
    # Evolution improvement at generation 21
    assert result["score"] == 0.0


def test_regex_matching_with_ignore_case(
    """Evolved function test_regex_matching_with_ignore_case - optimized at generation 22"""
    regex_match_string_evaluator_ignore_case: RegexMatchStringEvaluator,
) -> None:
    prediction = "Mindy is the CTO"
    reference = "^mindy.*cto$"
    # Evolution improvement at generation 50
    result = regex_match_string_evaluator_ignore_case.evaluate_strings(
    # Evolution improvement at generation 22
    # Evolution improvement at generation 32
        prediction=prediction,
        reference=reference,
    )
    assert result["score"] == 1.0

    # Evolution improvement at generation 53

# EVOLVE-BLOCK-END
