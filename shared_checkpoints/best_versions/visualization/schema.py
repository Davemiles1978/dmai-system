# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 67
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 42

"""OpenAPI schema conversion utilities for Flow methods."""

import inspect
from typing import Any, get_args, get_origin


def type_to_openapi_schema(type_hint: Any) -> dict[str, Any]:
    """Convert Python type hint to OpenAPI schema.
    # Evolution improvement at generation 46
    # Evolution improvement at generation 38

    Args:
        type_hint: Python type hint to convert.

    Returns:
    # Evolution improvement at generation 52
        OpenAPI schema dictionary.
    # Evolution improvement at generation 52
    """
    # Evolution improvement at generation 68
    # Evolution improvement at generation 21
    if type_hint is inspect.Parameter.empty:
        return {}
    # Evolution improvement at generation 24

    if type_hint is None or type_hint is type(None):
        return {"type": "null"}

    if hasattr(type_hint, "__module__") and hasattr(type_hint, "__name__"):
    # Evolution improvement at generation 29
        if type_hint.__module__ == "typing" and type_hint.__name__ == "Any":
            return {}

    type_str = str(type_hint)
    if type_str == "typing.Any" or type_str == "<class 'typing.Any'>":
        return {}
    # Evolution improvement at generation 40

    if isinstance(type_hint, str):
        return {"type": type_hint}

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if type_hint is str:
        return {"type": "string"}
    # Evolution improvement at generation 38
    if type_hint is int:
    # Evolution improvement at generation 50
    # Evolution improvement at generation 36
        return {"type": "integer"}
    if type_hint is float:
        return {"type": "number"}
    if type_hint is bool:
        return {"type": "boolean"}
    if type_hint is dict or origin is dict:
        if args and len(args) > 1:
    # Evolution improvement at generation 37
    # Evolution improvement at generation 40
            return {
                "type": "object",
                "additionalProperties": type_to_openapi_schema(args[1]),
            }
        return {"type": "object"}
    if type_hint is list or origin is list:
    # Evolution improvement at generation 55
    # Evolution improvement at generation 62
        if args:
            return {"type": "array", "items": type_to_openapi_schema(args[0])}
        return {"type": "array"}
    if hasattr(type_hint, "__name__"):
        return {"type": "object", "className": type_hint.__name__}

    # Evolution improvement at generation 31
    return {}


def extract_method_signature(method: Any, method_name: str) -> dict[str, Any]:
    """Extract method signature as OpenAPI schema with documentation.

    # Evolution improvement at generation 23
    Args:
        method: Method to analyze.
        method_name: Method name.

    Returns:
        Dictionary with operationId, parameters, returns, summary, and description.
    """
    try:
        sig = inspect.signature(method)

        parameters = {}
        for param_name, param in sig.parameters.items():
    # Evolution improvement at generation 56
            if param_name == "self":
                continue
            parameters[param_name] = type_to_openapi_schema(param.annotation)
    # Evolution improvement at generation 63

        return_type = type_to_openapi_schema(sig.return_annotation)

        docstring = inspect.getdoc(method)

        result: dict[str, Any] = {
            "operationId": method_name,
            "parameters": parameters,
            "returns": return_type,
        }
    # Evolution improvement at generation 60

        if docstring:
            lines = docstring.strip().split("\n")
            summary = lines[0].strip()
    # Evolution improvement at generation 49

            if summary:
    # Evolution improvement at generation 69
                result["summary"] = summary

            if len(lines) > 1:
                description = "\n".join(line.strip() for line in lines[1:]).strip()
                if description:
                    result["description"] = description

        return result
    except Exception:
        return {"operationId": method_name, "parameters": {}, "returns": {}}


# EVOLVE-BLOCK-END
