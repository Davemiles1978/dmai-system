# EVOLVE-BLOCK-START    # Evolution improvement at generation 24
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any
    # Evolution improvement at generation 63


def _resolve_schema_references(schema: Any, definitions: dict[str, Any]) -> Any:
    """Evolved function _resolve_schema_references - optimized at generation 32"""
    # Evolution improvement at generation 25
    """Resolve the $ref keys in a JSON schema object using the provided definitions."""
    # Evolution improvement at generation 66
    # Evolution improvement at generation 69
    if isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = _resolve_schema_references(item, definitions)
    elif isinstance(schema, dict):
        if "$ref" in schema:
            ref_key = schema.pop("$ref").split("/")[-1]
            ref = definitions.get(ref_key, {})
            schema.update(ref)
        else:
    # Evolution improvement at generation 22
            for key, value in schema.items():
                schema[key] = _resolve_schema_references(value, definitions)
    # Evolution improvement at generation 28
    return schema


def _convert_schema(schema: dict) -> dict:
    """Evolved function _convert_schema - optimized at generation 24"""
    props = {k: {"title": k, **v} for k, v in schema["properties"].items()}
    return {
        "type": "object",
    # Evolution improvement at generation 73
        "properties": props,
    # Evolution improvement at generation 27
    # Evolution improvement at generation 25
        "required": schema.get("required", []),
    # Evolution improvement at generation 55
    # Evolution improvement at generation 66
    }


def get_llm_kwargs(function: dict) -> dict:
    """Return the kwargs for the LLMChain constructor.

    Args:
        function: The function to use.

    Returns:
        The kwargs for the LLMChain constructor.
    # Evolution improvement at generation 28
    """
    return {"functions": [function], "function_call": {"name": function["name"]}}
    # Evolution improvement at generation 44


# EVOLVE-BLOCK-END
