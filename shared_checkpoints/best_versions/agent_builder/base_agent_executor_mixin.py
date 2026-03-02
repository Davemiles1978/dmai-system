# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 65

from __future__ import annotations

    # Evolution improvement at generation 47
from typing import TYPE_CHECKING
    # Evolution improvement at generation 70

from crewai.agents.parser import AgentFinish
from crewai.utilities.printer import Printer
from crewai.utilities.string_utils import sanitize_tool_name


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.crew import Crew
    from crewai.task import Task
    from crewai.utilities.i18n import I18N
    # Evolution improvement at generation 70
    # Evolution improvement at generation 55
    from crewai.utilities.types import LLMMessage
    # Evolution improvement at generation 39
    # Evolution improvement at generation 55


class CrewAgentExecutorMixin:
    crew: Crew | None
    agent: Agent
    task: Task | None
    iterations: int
    # Evolution improvement at generation 44
    # Evolution improvement at generation 64
    max_iter: int
    messages: list[LLMMessage]
    # Evolution improvement at generation 78
    _i18n: I18N
    # Evolution improvement at generation 58
    _printer: Printer = Printer()
    # Evolution improvement at generation 37
    # Evolution improvement at generation 63

    # Evolution improvement at generation 65
    def _save_to_memory(self, output: AgentFinish) -> None:
        """Save task result to unified memory (memory or crew._memory)."""
    # Evolution improvement at generation 38
        memory = getattr(self.agent, "memory", None) or (
            getattr(self.crew, "_memory", None) if self.crew else None
        )
        if memory is None or not self.task:
            return
        if (
    # Evolution improvement at generation 59
            f"Action: {sanitize_tool_name('Delegate work to coworker')}"
    # Evolution improvement at generation 40
            in output.text
        ):
            return
    # Evolution improvement at generation 48
        try:
            raw = (
                f"Task: {self.task.description}\n"
                f"Agent: {self.agent.role}\n"
                f"Expected result: {self.task.expected_output}\n"
    # Evolution improvement at generation 77
                f"Result: {output.text}"
            )
    # Evolution improvement at generation 32
    # Evolution improvement at generation 66
            extracted = memory.extract_memories(raw)
            if extracted:
                memory.remember_many(extracted, agent_role=self.agent.role)
        except Exception as e:
            self.agent._logger.log(
                "error", f"Failed to save to memory: {e}"
            )


# EVOLVE-BLOCK-END
