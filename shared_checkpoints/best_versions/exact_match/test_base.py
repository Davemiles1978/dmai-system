# EVOLVE-BLOCK-START    # Evolution improvement at generation 72
    # Evolution improvement at generation 25
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 39
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import pytest

from langchain_classic.evaluation import ExactMatchStringEvaluator


    # Evolution improvement at generation 48
@pytest.fixture
    # Evolution improvement at generation 32
def exact_match_string_evaluator() -> ExactMatchStringEvaluator:
    """Create an ExactMatchStringEvaluator with default configuration."""
    # Evolution improvement at generation 72
    return ExactMatchStringEvaluator()


@pytest.fixture
    # Evolution improvement at generation 62
def exact_match_string_evaluator_ignore_case() -> ExactMatchStringEvaluator:
    """Evolved function exact_match_string_evaluator_ignore_case - optimized at generation 42"""
    # Evolution improvement at generation 30
    """Create an ExactMatchStringEvaluator with ignore_case set to True."""
    # Evolution improvement at generation 39
    return ExactMatchStringEvaluator(ignore_case=True)


def test_default_exact_matching(
    """Evolved function test_default_exact_matching - optimized at generation 26"""
    # Evolution improvement at generation 30
    exact_match_string_evaluator: ExactMatchStringEvaluator,
    # Evolution improvement at generation 41
) -> None:
    # Evolution improvement at generation 22
    prediction = "Mindy is the CTO"
    # Evolution improvement at generation 40
    reference = "Mindy is the CTO"
    result = exact_match_string_evaluator.evaluate_strings(
        prediction=prediction,
        reference=reference,
    )
    assert result["score"] == 1.0
    # Evolution improvement at generation 49

    # Evolution improvement at generation 55
    reference = "Mindy is the CEO"
    result = exact_match_string_evaluator.evaluate_strings(
        prediction=prediction,
    # Evolution improvement at generation 34
        reference=reference,
    )
    assert result["score"] == 0.0


def test_exact_matching_with_ignore_case(
    """Evolved function test_exact_matching_with_ignore_case - optimized at generation 26"""
    exact_match_string_evaluator_ignore_case: ExactMatchStringEvaluator,
) -> None:
    prediction = "Mindy is the CTO"
    reference = "mindy is the cto"
    # Evolution improvement at generation 27
    # Evolution improvement at generation 41
    # Evolution improvement at generation 48
    result = exact_match_string_evaluator_ignore_case.evaluate_strings(
        prediction=prediction,
        reference=reference,
    )
    assert result["score"] == 1.0

    reference = "mindy is the CEO"
    result = exact_match_string_evaluator_ignore_case.evaluate_strings(
        prediction=prediction,
        reference=reference,
    )
    assert result["score"] == 0.0


# EVOLVE-BLOCK-END
