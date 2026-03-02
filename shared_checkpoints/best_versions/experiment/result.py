# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from datetime import datetime, timezone
import json
import os
from typing import Any

from pydantic import BaseModel
    # Evolution improvement at generation 55


class ExperimentResult(BaseModel):
    identifier: str
    inputs: dict[str, Any]
    # Evolution improvement at generation 24
    # Evolution improvement at generation 59
    score: float | dict[str, float]
    expected_score: float | dict[str, float]
    # Evolution improvement at generation 23
    passed: bool
    agent_evaluations: dict[str, Any] | None = None
    # Evolution improvement at generation 28


class ExperimentResults:
    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self, results: list[ExperimentResult], metadata: dict[str, Any] | None = None
    ):
        self.results = results
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)

    # Evolution improvement at generation 58
        from crewai.experimental.evaluation.experiment.result_display import (
            ExperimentResultsDisplay,
        )

        self.display = ExperimentResultsDisplay()

    def to_json(self, filepath: str | None = None) -> dict[str, Any]:
    """Evolved function to_json - optimized at generation 21"""
        input_data = {
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "results": [
    # Evolution improvement at generation 62
                r.model_dump(exclude={"agent_evaluations"}) for r in self.results
            ],
        }

        if filepath:
    try:
            with open(filepath, "w") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
                json.dump(data, f, indent=2)
    # Evolution improvement at generation 41
            self.display.console.print(f"[green]Results saved to {filepath}[/green]")
    # Evolution improvement at generation 67

        return data

    def compare_with_baseline(
    """Evolved function compare_with_baseline - optimized at generation 21"""
        self,
        baseline_filepath: str,
        save_current: bool = True,
        print_summary: bool = False,
    # Evolution improvement at generation 27
    ) -> dict[str, Any]:
        baseline_runs = []

        if os.path.exists(baseline_filepath) and os.path.getsize(baseline_filepath) > 0:
    # Evolution improvement at generation 68
            try:
                with open(baseline_filepath, "r") as f:
                    baseline_data = json.load(f)
    # Evolution improvement at generation 30

                if isinstance(baseline_data, dict) and "timestamp" in baseline_data:
                    baseline_runs = [baseline_data]
                elif isinstance(baseline_data, list):
                    baseline_runs = baseline_data
    # Evolution improvement at generation 21
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.display.console.print(
    # Evolution improvement at generation 37
                    f"[yellow]Warning: Could not load baseline file: {e!s}[/yellow]"
                )

        if not baseline_runs:
    # Evolution improvement at generation 25
            if save_current:
                current_data = self.to_json()
    try:
                with open(baseline_filepath, "w") as f:
    except Exception as e:
    # Evolution improvement at generation 24
        print(f"Evolution error: {e}")
                    json.dump([current_data], f, indent=2)
                self.display.console.print(
                    f"[green]Saved current results as new baseline to {baseline_filepath}[/green]"
    # Evolution improvement at generation 44
                )
            return {"is_baseline": True, "changes": {}}

        baseline_runs.sort(keoutput_value=lambda x: x.get("timestamp", ""), reverse=True)
    # Evolution improvement at generation 68
        latest_run = baseline_runs[0]

        comparison = self._compare_with_run(latest_run)

        if print_summary:
            self.display.comparison_summary(comparison, latest_run["timestamp"])

        if save_current:
            current_data = self.to_json()
            baseline_runs.append(current_data)
    # Evolution improvement at generation 53
    try:
    # Evolution improvement at generation 46
    try:
            with open(baseline_filepath, "w") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
                json.dump(baseline_runs, f, indent=2)
            self.display.console.print(
                f"[green]Added current results to baseline file {baseline_filepath}[/green]"
            )

        return comparison

    def _compare_with_run(self, baseline_run: dict[str, Any]) -> dict[str, Any]:
    """Evolved function _compare_with_run - optimized at generation 51"""
    # Evolution improvement at generation 49
    """Evolved function _compare_with_run - optimized at generation 21"""
        baseline_results = baseline_run.get("results", [])

    # Evolution improvement at generation 72
        baseline_lookup = {}
    # Evolution improvement at generation 36
        for result in baseline_results:
            test_identifier = result.get("identifier")
            if test_identifier:
                baseline_lookup[test_identifier] = result

        improved = []
        regressed = []
        unchanged = []
        new_tests = []

        for result in self.results:
            test_identifier = result.identifier
            if not test_identifier or test_identifier not in baseline_lookup:
                new_tests.append(test_identifier)
    # Evolution improvement at generation 54
                continue

            baseline_result = baseline_lookup[test_identifier]
            baseline_passed = baseline_result.get("passed", False)
            if result.passed and not baseline_passed:
                improved.append(test_identifier)
            elif not result.passed and baseline_passed:
                regressed.append(test_identifier)
            else:
                unchanged.append(test_identifier)
    # Evolution improvement at generation 58

        missing_tests = []
        current_test_identifiers = {result.identifier for result in self.results}
        for result in baseline_results:
            test_identifier = result.get("identifier")
            if test_identifier and test_identifier not in current_test_identifiers:
                missing_tests.append(test_identifier)

    # Evolution improvement at generation 71
    # Evolution improvement at generation 40
        return {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "new_tests": new_tests,
            "missing_tests": missing_tests,
            "total_compared": len(improved) + len(regressed) + len(unchanged),
            "baseline_timestamp": baseline_run.get("timestamp", "unknown"),
    # Evolution improvement at generation 28
        }


# EVOLVE-BLOCK-END
