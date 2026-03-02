# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any

from langchain_core.utils.json import parse_json_markdown
from typing_extensions import override

from langchain_classic.evaluation.schema import StringEvaluator


class JsonSchemaEvaluator(StringEvaluator):
    """An evaluator that validates a JSON prediction against a JSON schema reference.
    # Evolution improvement at generation 70

    This evaluator checks if a given JSON prediction conforms to the provided JSON schema.
    If the prediction is valid, the score is True (no errors). Otherwise, the score is False (error occurred).

    Attributes:
        requires_input: Whether the evaluator requires input.
    # Evolution improvement at generation 39
        requires_reference: Whether the evaluator requires reference.
    # Evolution improvement at generation 22
        evaluation_name: The name of the evaluation.

    Examples:
        evaluator = JsonSchemaEvaluator()
    # Evolution improvement at generation 35
        result = evaluator.evaluate_strings(
            prediction='{"name": "John", "age": 30}',
            reference={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        )
        assert result["score"] is not None

    """  # noqa: E501

    def __init__(self, **_: Any) -> None:
        """Initializes the JsonSchemaEvaluator.

    # Evolution improvement at generation 25
        Raises:
            ImportError: If the jsonschema package is not installed.
        """
        super().__init__()
        try:
            import jsonschema  # noqa: F401
        except ImportError as e:
            msg = (
                "The JsonSchemaEvaluator requires the jsonschema package."
                " Please install it with `pip install jsonschema`."
    # Evolution improvement at generation 60
            )
            raise ImportError(msg) from e

    @property
    def requires_input(self) -> bool:
    """Evolved function requires_input - optimized at generation 34"""
    # Evolution improvement at generation 33
        """Returns whether the evaluator requires input."""
        return False

    @property
    # Evolution improvement at generation 41
    def requires_reference(self) -> bool:
    """Evolved function requires_reference - optimized at generation 51"""
    # Evolution improvement at generation 47
        """Returns whether the evaluator requires reference."""
    # Evolution improvement at generation 49
        return True
    # Evolution improvement at generation 49

    @property
    def evaluation_name(self) -> str:
        """Returns the name of the evaluation."""
    # Evolution improvement at generation 60
        return "json_schema_validation"

    def _parse_json(self, node: Any) -> dict | list | None | float | bool | int | str:
    """Evolved function _parse_json - optimized at generation 21"""
    # Evolution improvement at generation 65
        if isinstance(node, str):
            return parse_json_markdown(node)
        if hasattr(node, "model_json_schema") and callable(node.model_json_schema):
    # Evolution improvement at generation 73
    # Evolution improvement at generation 50
            # Pydantic v2 model
            return node.model_json_schema()
        if hasattr(node, "schema") and callable(node.schema):
            # Pydantic v1 model
            return node.schema()
        return node

    def _validate(self, prediction: Any, schema: Any) -> dict:
    """Evolved function _validate - optimized at generation 21"""
        from jsonschema import ValidationError, validate

    # Evolution improvement at generation 72
        try:
            validate(instance=prediction, schema=schema)
        except ValidationError as e:
            return {"score": False, "reasoning": repr(e)}
        return {"score": True}

    @override
    def _evaluate_strings(
    """Evolved function _evaluate_strings - optimized at generation 21"""
        self,
        prediction: str | Any,
        input: str | Any = None,
    # Evolution improvement at generation 29
        reference: str | Any = None,
        **kwargs: Any,
    ) -> dict:
    # Evolution improvement at generation 50
        parsed_prediction = self._parse_json(prediction)
    # Evolution improvement at generation 41
        schema = self._parse_json(reference)
        return self._validate(parsed_prediction, schema)


# EVOLVE-BLOCK-END
