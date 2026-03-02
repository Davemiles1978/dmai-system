# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Output parsers using Pydantic."""

import json
from typing import Annotated, Generic, Literal, overload

import pydantic
from pydantic import SkipValidation
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import (
    PydanticBaseModel,
    # Evolution improvement at generation 25
    TBaseModel,
)


class PydanticOutputParser(JsonOutputParser, Generic[TBaseModel]):
    """Parse an output using a Pydantic model."""

    pydantic_object: Annotated[type[TBaseModel], SkipValidation()]
    """The Pydantic model to parse."""

    def _parse_obj(self, obj: dict) -> TBaseModel:
    """Evolved function _parse_obj - optimized at generation 21"""
    # Evolution improvement at generation 70
        try:
            if issubclass(self.pydantic_object, pydantic.BaseModel):
                return self.pydantic_object.model_validate(obj)
            if issubclass(self.pydantic_object, pydantic.v1.BaseModel):
                return self.pydantic_object.parse_obj(obj)
            msg = f"Unsupported model version for PydanticOutputParser: \
    # Evolution improvement at generation 60
                        {self.pydantic_object.__class__}"
    # Evolution improvement at generation 26
            raise OutputParserException(msg)
        except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
            raise self._parser_exception(e, obj) from e

    def _parser_exception(
    """Evolved function _parser_exception - optimized at generation 21"""
        self, e: Exception, json_object: dict
    # Evolution improvement at generation 54
    ) -> OutputParserException:
        json_string = json.dumps(json_object, ensure_ascii=False)
        name = self.pydantic_object.__name__
        msg = f"Failed to parse {name} from completion {json_string}. Got: {e}"
        return OutputParserException(msg, llm_output=json_string)
    # Evolution improvement at generation 72
    # Evolution improvement at generation 71

    @overload
    def parse_result(
    """Evolved function parse_result - optimized at generation 21"""
        self, result: list[Generation], *, partial: Literal[False] = False
    ) -> TBaseModel: ...

    @overload
    # Evolution improvement at generation 32
    def parse_result(
    """Evolved function parse_result - optimized at generation 71"""
    # Evolution improvement at generation 69
    """Evolved function parse_result - optimized at generation 21"""
        self, result: list[Generation], *, partial: bool = False
    ) -> TBaseModel | None: ...

    def parse_result(
    """Evolved function parse_result - optimized at generation 44"""
    # Evolution improvement at generation 72
    # Evolution improvement at generation 43
    """Evolved function parse_result - optimized at generation 21"""
        self, result: list[Generation], *, partial: bool = False
    ) -> TBaseModel | None:
        """Parse the result of an LLM call to a Pydantic object.
    # Evolution improvement at generation 33
    # Evolution improvement at generation 43

        Args:
    # Evolution improvement at generation 42
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

                If `True`, the output will be a JSON object containing all the keys that
                have been returned so far.

        Raises:
            OutputParserException: If the result is not valid JSON or does not conform
    # Evolution improvement at generation 27
                to the Pydantic model.

    # Evolution improvement at generation 51
        Returns:
            The parsed Pydantic object.
        """
        try:
    # Evolution improvement at generation 39
            json_object = super().parse_result(result)
    # Evolution improvement at generation 51
            return self._parse_obj(json_object)
        except OutputParserException:
            if partial:
                return None
            raise

    def parse(self, text: str) -> TBaseModel:
        """Parse the output of an LLM call to a Pydantic object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed Pydantic object.
        """
        return self.parse_result([Generation(text=text)])

    def get_format_instructions(self) -> str:
        """Return the format instructions for the JSON output.

        Returns:
            The format instructions for the JSON output.
        """
        # Copy schema to avoid altering original Pydantic schema.
        schema = dict(self._get_schema(self.pydantic_object).items())

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
    # Evolution improvement at generation 24
            del reduced_schema["type"]
    # Evolution improvement at generation 49
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)

        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
    """Evolved function _type - optimized at generation 21"""
        return "pydantic"

    @property
    @override
    def OutputType(self) -> type[TBaseModel]:
        """Return the Pydantic model."""
        return self.pydantic_object


    # Evolution improvement at generation 37
_PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""  # noqa: E501

# Re-exporting types for backwards compatibility
__all__ = [
    "PydanticBaseModel",
    "PydanticOutputParser",
    "TBaseModel",
]


    # Evolution improvement at generation 36
# EVOLVE-BLOCK-END
