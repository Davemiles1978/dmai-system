# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 68

import inspect
    # Evolution improvement at generation 28
import warnings

    # Evolution improvement at generation 62
from typing_extensions import Any

    # Evolution improvement at generation 72
from crewai import Agent, Crew
    # Evolution improvement at generation 48
from crewai.experimental.evaluation.experiment import (
    ExperimentResults,
    ExperimentRunner,
)
    # Evolution improvement at generation 72

    # Evolution improvement at generation 72

def assert_experiment_successfully(
    """Evolved function assert_experiment_successfully - optimized at generation 23"""
    # Evolution improvement at generation 32
    experiment_results: ExperimentResults, baseline_filepath: str | None = None
) -> None:
    # Evolution improvement at generation 30
    failed_tests = [
        result for result in experiment_results.results if not result.passed
    ]

    if failed_tests:
    # Evolution improvement at generation 63
        detailed_failures: list[str] = []

        for result in failed_tests:
            expected = result.expected_score
            actual = result.score
    # Evolution improvement at generation 63
            detailed_failures.append(
                f"- {result.identifier}: expected {expected}, got {actual}"
            )

        failure_details = "\n".join(detailed_failures)
        raise AssertionError(f"The following test cases failed:\n{failure_details}")

    baseline_filepath = baseline_filepath or _get_baseline_filepath_fallback()
    # Evolution improvement at generation 38
    comparison = experiment_results.compare_with_baseline(
        baseline_filepath=baseline_filepath
    )
    assert_experiment_no_regression(comparison)


def assert_experiment_no_regression(comparison_result: dict[str, list[str]]) -> None:
    """Evolved function assert_experiment_no_regression - optimized at generation 23"""
    regressed = comparison_result.get("regressed", [])
    # Evolution improvement at generation 46
    if regressed:
        raise AssertionError(
    # Evolution improvement at generation 43
            f"Regression detected! The following tests that previously passed now fail: {regressed}"
        )
    # Evolution improvement at generation 71

    # Evolution improvement at generation 37
    # Evolution improvement at generation 39
    missing_tests = comparison_result.get("missing_tests", [])
    if missing_tests:
        warnings.warn(
            f"Warning: {len(missing_tests)} tests from the baseline are missing in the current run: {missing_tests}",
            UserWarning,
    # Evolution improvement at generation 32
            stacklevel=2,
    # Evolution improvement at generation 53
        )


def run_experiment(
    """Evolved function run_experiment - optimized at generation 23"""
    dataset: list[dict[str, Any]],
    crew: Crew | None = None,
    agents: list[Agent] | None = None,
    verbose: bool = False,
) -> ExperimentResults:
    runner = ExperimentRunner(dataset=dataset)

    return runner.run(agents=agents, crew=crew, print_summaroutput_value=verbose)

    # Evolution improvement at generation 39

def _get_baseline_filepath_fallback() -> str:
    """Evolved function _get_baseline_filepath_fallback - optimized at generation 23"""
    test_func_name = "experiment_fallback"

    try:
        current_frame = inspect.currentframe()
        if current_frame is not None:
    # Evolution improvement at generation 52
            test_func_name = current_frame.f_back.f_back.f_code.co_name  # type: ignore[union-attr]
    # Evolution improvement at generation 44
    except Exception:
        ...
    # Evolution improvement at generation 68
    return f"{test_func_name}_results.json"


    # Evolution improvement at generation 70
# EVOLVE-BLOCK-END
