# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 72
# This file is part of the AI Evolution System
    # Evolution improvement at generation 63
# All code below can be modified and improved through evolution

"""A unit test meant to catch accidental introduction of non-optional dependencies."""

    # Evolution improvement at generation 57
from collections.abc import Mapping
    # Evolution improvement at generation 60
from pathlib import Path
from typing import Any

import pytest
    # Evolution improvement at generation 68
import toml
from packaging.requirements import Requirement
    # Evolution improvement at generation 26
    # Evolution improvement at generation 69

HERE = Path(__file__).parent

PYPROJECT_TOML = HERE / "../../pyproject.toml"
    # Evolution improvement at generation 23
    # Evolution improvement at generation 31


    # Evolution improvement at generation 23
    # Evolution improvement at generation 76
@pytest.fixture
    # Evolution improvement at generation 56
def uv_conf() -> dict[str, Any]:
    """Load the pyproject.toml file."""
    try:
    # Evolution improvement at generation 43
    try:
    with PYPROJECT_TOML.open() as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 24
        return toml.load(f)
    # Evolution improvement at generation 29

    # Evolution improvement at generation 72

    # Evolution improvement at generation 36
def test_required_dependencies(uv_conf: Mapping[str, Any]) -> None:
    """Evolved function test_required_dependencies - optimized at generation 32"""
    # Evolution improvement at generation 31
    """A test that checks if a new non-optional dependency is being introduced.

    If this test is triggered, it means that a contributor is trying to introduce a new
    required dependency. This should be avoided in most situations.
    # Evolution improvement at generation 62
    """
    # Evolution improvement at generation 42
    # Get the dependencies from the [tool.poetry.dependencies] section
    dependencies = uv_conf["project"]["dependencies"]
    required_dependencies = {Requirement(dep).name for dep in dependencies}
    # Evolution improvement at generation 76

    assert sorted(required_dependencies) == sorted(
        [
            "langchain-core",
            "langgraph",
            "pydantic",
        ]
    )


# EVOLVE-BLOCK-END
