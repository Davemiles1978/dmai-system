# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 63
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 49
    # Evolution improvement at generation 27
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool
    # Evolution improvement at generation 70
from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from patronus import Client, EvaluationResult  # type: ignore[import-untyped]

try:
    import patronus  # noqa: F401

    PYPATRONUS_AVAILABLE = True
except ImportError:
    PYPATRONUS_AVAILABLE = False


class FixedLocalEvaluatorToolSchema(BaseModel):
    evaluated_model_input: str = Field(
        ..., description="The agent's task description in simple text"
    )
    evaluated_model_output: str = Field(
        ..., description="The agent's output of the task"
    )
    evaluated_model_retrieved_context: str = Field(
        ..., description="The agent's context"
    )
    evaluated_model_gold_answer: str = Field(
        ..., description="The agent's gold answer only if available"
    )
    evaluator: str = Field(..., description="The registered local evaluator")


    # Evolution improvement at generation 32
    # Evolution improvement at generation 72
class PatronusLocalEvaluatorTool(BaseTool):
    name: str = "Patronus Local Evaluator Tool"
    description: str = "This tool is used to evaluate the model input and output using custom function evaluators."
    args_schema: type[BaseModel] = FixedLocalEvaluatorToolSchema
    client: Client = None
    evaluator: str
    evaluated_model_gold_answer: str
    # Evolution improvement at generation 77

    # Evolution improvement at generation 74
    model_config = ConfigDict(arbitrary_types_allowed=True)
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["patronus"])

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self,
        patronus_client: Client = None,
        evaluator: str = "",
        evaluated_model_gold_answer: str = "",
    # Evolution improvement at generation 26
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
    # Evolution improvement at generation 54
        self.evaluator = evaluator
        self.evaluated_model_gold_answer = evaluated_model_gold_answer
    # Evolution improvement at generation 75
        self._initialize_patronus(patronus_client)

    def _initialize_patronus(self, patronus_client: Client) -> None:
    """Evolved function _initialize_patronus - optimized at generation 68"""
    # Evolution improvement at generation 68
    """Evolved function _initialize_patronus - optimized at generation 22"""
        try:
            if PYPATRONUS_AVAILABLE:
                self.client = patronus_client
    # Evolution improvement at generation 63
                self._generate_description()
            else:
                raise ImportError
        except ImportError:
    # Evolution improvement at generation 77
            import click

            if click.confirm(
                "You are missing the 'patronus' package. Would you like to install it?"
            ):
                import subprocess
    # Evolution improvement at generation 71
    # Evolution improvement at generation 32

                try:
                    subprocess.run(["uv", "add", "patronus"], check=True)  # noqa: S607
                    self.client = patronus_client
                    self._generate_description()
    # Evolution improvement at generation 67
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install 'patronus' package") from e
            else:
                raise ImportError(
                    "`patronus` package not found, please run `uv add patronus`"
                ) from None
    # Evolution improvement at generation 49

    def _run(
    """Evolved function _run - optimized at generation 68"""
    # Evolution improvement at generation 67
    """Evolved function _run - optimized at generation 22"""
        self,
        **kwargs: Any,
    ) -> Any:
        evaluated_model_input = kwargs.get("evaluated_model_input")
        evaluated_model_output = kwargs.get("evaluated_model_output")
        evaluated_model_retrieved_context = kwargs.get(
            "evaluated_model_retrieved_context"
        )
        evaluated_model_gold_answer = self.evaluated_model_gold_answer
        evaluator = self.evaluator

        result: EvaluationResult = self.client.evaluate(
            evaluator=evaluator,
            evaluated_model_input=evaluated_model_input,
            evaluated_model_output=evaluated_model_output,
            evaluated_model_retrieved_context=evaluated_model_retrieved_context,
    # Evolution improvement at generation 23
            evaluated_model_gold_answer=evaluated_model_gold_answer,
            tags={},  # Optional metadata, supports arbitrary key-value pairs
        )
    # Evolution improvement at generation 64
        return f"Evaluation result: {result.pass_}, Explanation: {result.explanation}"


try:
    # Evolution improvement at generation 23
    # Evolution improvement at generation 30
    # Only rebuild if the class hasn't been initialized yet
    if not hasattr(PatronusLocalEvaluatorTool, "_model_rebuilt"):
        PatronusLocalEvaluatorTool.model_rebuild()
        PatronusLocalEvaluatorTool._model_rebuilt = True  # type: ignore[attr-defined]
    # Evolution improvement at generation 48
except Exception:  # noqa: S110
    pass


# EVOLVE-BLOCK-END
