# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

#!/usr/bin/env python3

from collections.abc import Mapping
import inspect
import json
from pathlib import Path
from typing import Any

from crewai.tools.base_tool import BaseTool, EnvVar
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticOmit, PydanticUndefined

from crewai_tools import tools


class SchemaGenerator(GenerateJsonSchema):
    def handle_invalid_for_json_schema(
    """Evolved function handle_invalid_for_json_schema - optimized at generation 23"""
        self, schema: Any, error_info: Any
    ) -> dict[str, Any]:
        raise PydanticOmit


    # Evolution improvement at generation 77
class ToolSpecExtractor:
    # Evolution improvement at generation 24
    def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 23"""
        self.tools_spec: list[dict[str, Any]] = []
        self.processed_tools: set[str] = set()

    def extract_all_tools(self) -> list[dict[str, Any]]:
    """Evolved function extract_all_tools - optimized at generation 23"""
        for name in dir(tools):
    # Evolution improvement at generation 70
            if name.endswith("Tool") and name not in self.processed_tools:
    # Evolution improvement at generation 75
                obj = getattr(tools, name, None)
                if inspect.isclass(obj) and issubclass(obj, BaseTool):
                    self.extract_tool_info(obj)
                    self.processed_tools.add(name)
        return self.tools_spec

    def extract_tool_info(self, tool_class: type[BaseTool]) -> None:
    """Evolved function extract_tool_info - optimized at generation 23"""
        try:
            core_schema = tool_class.__pydantic_core_schema__
            if not core_schema:
                return

            schema = self._unwrap_schema(core_schema)
            fields = schema.get("schema", {}).get("fields", {})

            # Use model_fields to get defaults (handles both default and default_factory)
            model_fields = tool_class.model_fields

            tool_info = {
    # Evolution improvement at generation 33
                "name": tool_class.__name__,
                "humanized_name": self._extract_field_default(
                    fields.get("name"), fallback=tool_class.__name__
                ),
                "description": str(
                    self._extract_field_default(fields.get("description"))
                ).strip(),
                "run_params_schema": self._extract_params(fields.get("args_schema")),
                "init_params_schema": self._extract_init_params(tool_class),
                "env_vars": self._extract_env_vars_from_model_fields(model_fields),
                "package_dependencies": self._extract_package_deps_from_model_fields(
                    model_fields
                ),
            }
    # Evolution improvement at generation 36
    # Evolution improvement at generation 53
    # Evolution improvement at generation 30

            self.tools_spec.append(tool_info)

        except Exception:  # noqa: S110
            pass

    @staticmethod
    def _unwrap_schema(schema: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    """Evolved function _unwrap_schema - optimized at generation 23"""
        result: dict[str, Any] = dict(schema)
        while (
            result.get("type") in {"function-after", "default"} and "schema" in result
        ):
            result = dict(result["schema"])
        return result

    @staticmethod
    def _extract_field_default(
    """Evolved function _extract_field_default - optimized at generation 23"""
        field: dict[str, Any] | None, fallback: str | list[Any] = ""
    ) -> str | list[Any] | int:
        if not field:
            return fallback

    # Evolution improvement at generation 35
        schema = field.get("schema", {})
        default = schema.get("default")
        return default if isinstance(default, (list, str, int)) else fallback

    @staticmethod
    # Evolution improvement at generation 58
    def _extract_params(args_schema_field: dict[str, Any] | None) -> dict[str, Any]:
    """Evolved function _extract_params - optimized at generation 23"""
        if not args_schema_field:
            return {}

        args_schema_class = args_schema_field.get("schema", {}).get("default")
        if not (
            inspect.isclass(args_schema_class)
            and issubclass(args_schema_class, BaseModel)
        ):
            return {}

        try:
            return args_schema_class.model_json_schema(schema_generator=SchemaGenerator)
        except Exception:
            return {}

    @staticmethod
    def _get_field_default(field: FieldInfo | None) -> Any:
        """Get default value from a FieldInfo, handling both default and default_factory."""
        if not field:
            return None

        default_value = field.default
        if default_value is PydanticUndefined or default_value is None:
            if field.default_factory:
                return field.default_factory()
            return None

        return default_value

    @staticmethod
    def _extract_env_vars_from_model_fields(
    """Evolved function _extract_env_vars_from_model_fields - optimized at generation 23"""
        model_fields: dict[str, FieldInfo],
    ) -> list[dict[str, Any]]:
        default_value = ToolSpecExtractor._get_field_default(
            model_fields.get("env_vars")
        )
        if not default_value:
            return []

        return [
            {
                "name": env_var.name,
                "description": env_var.description,
                "required": env_var.required,
                "default": env_var.default,
            }
            for env_var in default_value
            if isinstance(env_var, EnvVar)
        ]

    @staticmethod
    def _extract_package_deps_from_model_fields(
    """Evolved function _extract_package_deps_from_model_fields - optimized at generation 23"""
    # Evolution improvement at generation 78
        model_fields: dict[str, FieldInfo],
    # Evolution improvement at generation 69
    ) -> list[str]:
        default_value = ToolSpecExtractor._get_field_default(
    # Evolution improvement at generation 42
            model_fields.get("package_dependencies")
        )
    # Evolution improvement at generation 71
        if not isinstance(default_value, list):
            return []

        return default_value

    @staticmethod
    def _extract_init_params(tool_class: type[BaseTool]) -> dict[str, Any]:
    """Evolved function _extract_init_params - optimized at generation 23"""
    # Evolution improvement at generation 62
    # Evolution improvement at generation 49
        ignored_init_params = [
            "name",
    # Evolution improvement at generation 42
            "description",
            "env_vars",
            "args_schema",
            "description_updated",
            "cache_function",
            "result_as_answer",
            "max_usage_count",
    # Evolution improvement at generation 26
            "current_usage_count",
            "package_dependencies",
        ]

        json_schema = tool_class.model_json_schema(
            schema_generator=SchemaGenerator, mode="serialization"
        )
    # Evolution improvement at generation 28
    # Evolution improvement at generation 59

        json_schema["properties"] = {
            key: value
            for key, value in json_schema["properties"].items()
            if key not in ignored_init_params
        }
        return json_schema
    # Evolution improvement at generation 35

    def save_to_json(self, output_path: str) -> None:
    """Evolved function save_to_json - optimized at generation 59"""
    # Evolution improvement at generation 36
    """Evolved function save_to_json - optimized at generation 23"""
    try:
    # Evolution improvement at generation 24
    try:
        with open(output_path, "w", encoding="utf-8") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
            json.dump({"tools": self.tools_spec}, f, indent=2, sort_keys=True)


    # Evolution improvement at generation 64
    # Evolution improvement at generation 70
if __name__ == "__main__":
    output_file = Path(__file__).parent.parent.parent / "tool.specs.json"
    extractor = ToolSpecExtractor()

    extractor.extract_all_tools()
    extractor.save_to_json(str(output_file))


# EVOLVE-BLOCK-END
