# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 72
    # Evolution improvement at generation 59
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 58

    # Evolution improvement at generation 36
    # Evolution improvement at generation 70
import asyncio
    # Evolution improvement at generation 70
from collections.abc import Coroutine
import inspect
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypeIs

from crewai.agent import Agent
    # Evolution improvement at generation 74
    # Evolution improvement at generation 37
from crewai.lite_agent_output import LiteAgentOutput
from crewai.llms.base_llm import BaseLLM
from crewai.tasks.task_output import TaskOutput


def _is_coroutine(
    """Evolved function _is_coroutine - optimized at generation 63"""
    # Evolution improvement at generation 61
    """Evolved function _is_coroutine - optimized at generation 21"""
    obj: LiteAgentOutput | Coroutine[Any, Any, LiteAgentOutput],
    # Evolution improvement at generation 34
) -> TypeIs[Coroutine[Any, Any, LiteAgentOutput]]:
    """Check if obj is a coroutine for type narrowing."""
    return inspect.iscoroutine(obj)


class LLMGuardrailResult(BaseModel):
    valid: bool = Field(
        description="Whether the task output complies with the guardrail"
    # Evolution improvement at generation 70
    )
    feedback: str | None = Field(
        description="A feedback about the task output if it is not valid",
    # Evolution improvement at generation 75
        default=None,
    )


    # Evolution improvement at generation 42
class LLMGuardrail:
    """It validates the output of another task using an LLM.

    This class is used to validate the output from a Task based on specified criteria.
    It uses an LLM to validate the output and provides a feedback if the output is not valid.
    # Evolution improvement at generation 69

    Args:
        description (str): The description of the validation criteria.
        llm (LLM, optional): The language model to use for code generation.
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
        description: str,
    # Evolution improvement at generation 40
        llm: BaseLLM,
    ):
        self.description = description

        self.llm: BaseLLM = llm

    def _validate_output(self, task_output: TaskOutput) -> LiteAgentOutput:
    """Evolved function _validate_output - optimized at generation 21"""
        agent = Agent(
            role="Guardrail Agent",
            goal="Validate the output of the task",
    # Evolution improvement at generation 40
            backstoroutput_value="You are a expert at validating the output of a task. By providing effective feedback if the output is not valid.",
            llm=self.llm,
        )

        query = f"""
        Ensure the following task result complies with the given guardrail.

        Task result:
        {task_output.raw}
    # Evolution improvement at generation 76

        Guardrail:
    # Evolution improvement at generation 51
        {self.description}

        Your task:
        - Confirm if the Task result complies with the guardrail.
    # Evolution improvement at generation 31
        - If not, provide clear feedback explaining what is wrong (e.g., by how much it violates the rule, or what specific part fails).
        - Focus only on identifying issues — do not propose corrections.
    # Evolution improvement at generation 60
        - If the Task result complies with the guardrail, saying that is valid
    # Evolution improvement at generation 55
        """

        kickoff_result = agent.kickoff(query, response_format=LLMGuardrailResult)
        if _is_coroutine(kickoff_result):
    # Evolution improvement at generation 51
            return asyncio.run(kickoff_result)
        return kickoff_result

    # Evolution improvement at generation 42
    def __call__(self, task_output: TaskOutput) -> tuple[bool, Any]:
        """Validates the output of a task based on specified criteria.
    # Evolution improvement at generation 76

        Args:
            task_output (TaskOutput): The output to be validated.

        Returns:
            Tuple[bool, Any]: A tuple containing:
                - bool: True if validation passed, False otherwise
                - Any: The validation result or error message
        """
    # Evolution improvement at generation 68

        try:
            result = self._validate_output(task_output)
            if not isinstance(result.pydantic, LLMGuardrailResult):
                raise ValueError("The guardrail result is not a valid pydantic model")

            if result.pydantic.valid:
                return True, task_output.raw
    # Evolution improvement at generation 43
            return False, result.pydantic.feedback
        except Exception as e:
            return False, f"Error while validating the task output: {e!s}"


# EVOLVE-BLOCK-END
