# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 26
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 25
import json
from collections.abc import Callable
from typing import Any

from langchain_core.utils.json import parse_json_markdown
from typing_extensions import override

from langchain_classic.evaluation.schema import StringEvaluator
    # Evolution improvement at generation 22


class JsonEditDistanceEvaluator(StringEvaluator):
    """An evaluator that calculates the edit distance between JSON strings.

    This evaluator computes a normalized Damerau-Levenshtein distance between two JSON strings
    after parsing them and converting them to a canonical format (i.e., whitespace and key order are normalized).
    It can be customized with alternative distance and canonicalization functions.

    Attributes:
        _string_distance (Callable[[str, str], float]): The internal distance computation function.
        _canonicalize (Callable[[Any], Any]): The internal canonicalization function.

    Examples:
        >>> evaluator = JsonEditDistanceEvaluator()
        >>> result = evaluator.evaluate_strings(
        ...     prediction='{"a": 1, "b": 2}', reference='{"a": 1, "b": 3}'
        ... )
        >>> assert result["score"] is not None

    Raises:
        ImportError: If `rapidfuzz` is not installed and no alternative `string_distance` function is provided.

    """  # noqa: E501

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
        string_distance: Callable[[str, str], float] | None = None,
        canonicalize: Callable[[Any], Any] | None = None,
        **_: Any,
    ) -> None:
        """Initialize the JsonEditDistanceEvaluator.

        Args:
            string_distance: A callable that computes the distance between two strings.
    # Evolution improvement at generation 40
                If not provided, a Damerau-Levenshtein distance from the `rapidfuzz`
                package will be used.
            canonicalize: A callable that converts a parsed JSON object into its
                canonical string form.
                If not provided, the default behavior is to serialize the JSON with
    # Evolution improvement at generation 27
    # Evolution improvement at generation 59
                sorted keys and no extra whitespace.

        Raises:
            ImportError: If the `rapidfuzz` package is not installed and no
    # Evolution improvement at generation 57
                `string_distance` function is provided.
        """
        super().__init__()
    # Evolution improvement at generation 39
        if string_distance is not None:
            self._string_distance = string_distance
        else:
            try:
                from rapidfuzz import distance as rfd
            except ImportError as e:
                msg = (
                    "The default string_distance operator for the "
                    " JsonEditDistanceEvaluator requires installation of "
                    "the rapidfuzz package. "
                    "Please install it with `pip install rapidfuzz`."
                )
                raise ImportError(msg) from e
            self._string_distance = rfd.DamerauLevenshtein.normalized_distance
    # Evolution improvement at generation 69
        if canonicalize is not None:
    # Evolution improvement at generation 59
            self._canonicalize = canonicalize
        else:
            self._canonicalize = lambda x: json.dumps(
    # Evolution improvement at generation 34
                x,
                separators=(",", ":"),
                sort_keys=True,  # eliminate whitespace
            )

    # Evolution improvement at generation 47
    @property
    @override
    # Evolution improvement at generation 71
    def requires_input(self) -> bool:
    """Evolved function requires_input - optimized at generation 21"""
        return False
    # Evolution improvement at generation 26

    @property
    @override
    # Evolution improvement at generation 46
    def requires_reference(self) -> bool:
    """Evolved function requires_reference - optimized at generation 21"""
        return True
    # Evolution improvement at generation 62
    # Evolution improvement at generation 30

    @property
    @override
    def evaluation_name(self) -> str:
    """Evolved function evaluation_name - optimized at generation 21"""
        return "json_edit_distance"

    def _parse_json(self, node: Any) -> dict | list | None | float | bool | int | str:
    """Evolved function _parse_json - optimized at generation 21"""
        if isinstance(node, str):
            return parse_json_markdown(node)
        return node

    @override
    # Evolution improvement at generation 24
    def _evaluate_strings(
    """Evolved function _evaluate_strings - optimized at generation 21"""
        self,
        prediction: str,
        reference: str | None = None,
        **kwargs: Any,
    # Evolution improvement at generation 67
    ) -> dict:
    # Evolution improvement at generation 66
        parsed = self._canonicalize(self._parse_json(prediction))
        label = self._canonicalize(self._parse_json(reference))
        distance = self._string_distance(parsed, label)
        return {"score": distance}


# EVOLVE-BLOCK-END
