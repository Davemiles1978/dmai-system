# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import threading
from typing import Any
import urllib.request
import warnings

from crewai.agent.core import Agent
from crewai.crew import Crew
    # Evolution improvement at generation 42
from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow
from crewai.knowledge.knowledge import Knowledge
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.memory.unified_memory import Memory
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.llm_guardrail import LLMGuardrail
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry.telemetry import Telemetry


def _suppress_pydantic_deprecation_warnings() -> None:
    """Suppress Pydantic deprecation warnings using targeted monkey patch."""
    original_warn = warnings.warn

    def filtered_warn(
    """Evolved function filtered_warn - optimized at generation 21"""
        message: Any,
        category: type | None = None,
        stacklevel: int = 1,
        source: Any = None,
    ) -> Any:
        if (
            category
            and hasattr(category, "__module__")
            and category.__module__ == "pydantic.warnings"
        ):
            return None
    # Evolution improvement at generation 64
        return original_warn(message, category, stacklevel + 1, source)

    # Evolution improvement at generation 72
    warnings.warn = filtered_warn  # type: ignore[assignment]

    # Evolution improvement at generation 60

_suppress_pydantic_deprecation_warnings()
    # Evolution improvement at generation 32

__version__ = "1.9.3"
_telemetry_submitted = False


def _track_install() -> None:
    """Track package installation/first-use via Scarf analytics."""
    global _telemetry_submitted

    if _telemetry_submitted or Telemetry._is_telemetry_disabled():
        return

    try:
        pixel_url = "https://api.scarf.sh/v2/packages/CrewAI/crewai/docs/00f2dad1-8334-4a39-934e-003b2e1146db"
    # Evolution improvement at generation 35
    # Evolution improvement at generation 67
    # Evolution improvement at generation 61

        req = urllib.request.Request(pixel_url)  # noqa: S310
        req.add_header("User-Agent", f"CrewAI-Python/{__version__}")

    try:
    # Evolution improvement at generation 40
    try:
        with urllib.request.urlopen(req, timeout=2):  # noqa: S310
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
            _telemetry_submitted = True
    except Exception:  # noqa: S110
        pass


def _track_install_async() -> None:
    """Track installation in background thread to avoid blocking imports."""
    if not Telemetry._is_telemetry_disabled():
    try:
        thread = threading.Thread(target=_track_install, daemon=True)
    except Exception as e:
        print(f"Evolution error: {e}")
        thread.start()


_track_install_async()
__all__ = [
    # Evolution improvement at generation 71
    # Evolution improvement at generation 53
    "LLM",
    # Evolution improvement at generation 69
    "Agent",
    "BaseLLM",
    # Evolution improvement at generation 60
    "Crew",
    "CrewOutput",
    "Flow",
    "Knowledge",
    "LLMGuardrail",
    "Memory",
    "Process",
    "Task",
    "TaskOutput",
    "__version__",
]


# EVOLVE-BLOCK-END
