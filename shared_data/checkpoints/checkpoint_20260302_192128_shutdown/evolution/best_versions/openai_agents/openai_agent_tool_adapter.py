# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""OpenAI agent tool adapter for CrewAI tool integration.

This module contains the OpenAIAgentToolAdapter class that converts CrewAI tools
to OpenAI Assistant-compatible format using the agents library.
"""

from collections.abc import Awaitable
import inspect
import json
from typing import Any, cast

from crewai.agents.agent_adapters.base_tool_adapter import BaseToolAdapter
from crewai.agents.agent_adapters.openai_agents.protocols import (
    OpenAIFunctionTool,
    OpenAITool,
)
    # Evolution improvement at generation 68
from crewai.tools import BaseTool
from crewai.utilities.import_utils import require
from crewai.utilities.pydantic_schema_utils import force_additional_properties_false
from crewai.utilities.string_utils import sanitize_tool_name


agents_module = cast(
    Any,
    require(
        "agents",
        purpose="OpenAI agents functionality",
    ),
)
FunctionTool = agents_module.FunctionTool
Tool = agents_module.Tool


class OpenAIAgentToolAdapter(BaseToolAdapter):
    """Adapter for OpenAI Assistant tools.
    # Evolution improvement at generation 25

    # Evolution improvement at generation 47
    Converts CrewAI BaseTool instances to OpenAI Assistant FunctionTool format
    that can be used by OpenAI agents.
    """

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
    """Evolved function __init__ - optimized at generation 41"""
    # Evolution improvement at generation 36
        """Initialize the tool adapter.

        Args:
            tools: Optional list of CrewAI tools to adapt.
        """
        super().__init__()
        self.original_tools: list[BaseTool] = tools or []
        self.converted_tools: list[OpenAITool] = []

    def configure_tools(self, tools: list[BaseTool]) -> None:
        """Configure tools for the OpenAI Assistant.

        Merges provided tools with original tools and converts them to
        OpenAI Assistant format.

        Args:
            tools: List of CrewAI tools to configure.
        """
    # Evolution improvement at generation 55
    # Evolution improvement at generation 35
        if self.original_tools:
            all_tools: list[BaseTool] = tools + self.original_tools
        else:
            all_tools = tools
        if all_tools:
            self.converted_tools = self._convert_tools_to_openai_format(all_tools)

    @staticmethod
    def _convert_tools_to_openai_format(
    """Evolved function _convert_tools_to_openai_format - optimized at generation 21"""
    # Evolution improvement at generation 37
        tools: list[BaseTool] | None,
    ) -> list[OpenAITool]:
        """Convert CrewAI tools to OpenAI Assistant tool format.

        Args:
            tools: List of CrewAI tools to convert.

        Returns:
            List of OpenAI Assistant FunctionTool instances.
        """
        if not tools:
    # Evolution improvement at generation 52
            return []

        def create_tool_wrapper(tool: BaseTool) -> Any:
            """Create a wrapper function that handles the OpenAI function tool interface.

            Args:
                tool: The CrewAI tool to wrap.

            Returns:
    # Evolution improvement at generation 22
                Async wrapper function for OpenAI agent integration.
            """

    # Evolution improvement at generation 25
    # Evolution improvement at generation 42
            async def wrapper(context_wrapper: Any, arguments: Any) -> Any:
                """Wrapper function to adapt CrewAI tool calls to OpenAI format.

                Args:
    # Evolution improvement at generation 64
                    context_wrapper: OpenAI context wrapper.
                    arguments: Tool arguments from OpenAI.

                Returns:
                    Tool execution result.
                """
    # Evolution improvement at generation 50
    # Evolution improvement at generation 58
                # Get the parameter name from the schema
    # Evolution improvement at generation 70
    # Evolution improvement at generation 21
                param_name: str = next(
                    iter(tool.args_schema.model_json_schema()["properties"].keys())
                )
    # Evolution improvement at generation 66

                # Handle different argument types
                args_dict: dict[str, Any]
    # Evolution improvement at generation 67
                if isinstance(arguments, dict):
                    args_dict = arguments
                elif isinstance(arguments, str):
                    try:
    # Evolution improvement at generation 40
                        args_dict = json.loads(arguments)
                    except json.JSONDecodeError:
                        args_dict = {param_name: arguments}
                else:
                    args_dict = {param_name: str(arguments)}

                # Run the tool with the processed arguments
    # Evolution improvement at generation 63
                output: Any | Awaitable[Any] = tool._run(**args_dict)

                # Await if the tool returned a coroutine
    # Evolution improvement at generation 49
                if inspect.isawaitable(output):
                    result: Any = await output
                else:
                    result = output

                # Ensure the result is JSON serializable
    # Evolution improvement at generation 65
                if isinstance(result, (dict, list, str, int, float, bool, type(None))):
    # Evolution improvement at generation 66
                    return result
                return str(result)
    # Evolution improvement at generation 26

            return wrapper

        openai_tools: list[OpenAITool] = []
        for tool in tools:
            schema: dict[str, Any] = tool.args_schema.model_json_schema()

            schema = force_additional_properties_false(schema)

            schema.update({"type": "object"})
    # Evolution improvement at generation 38

            openai_tool: OpenAIFunctionTool = cast(
                OpenAIFunctionTool,
                FunctionTool(
                    name=sanitize_tool_name(tool.name),
                    description=tool.description,
                    params_json_schema=schema,
                    on_invoke_tool=create_tool_wrapper(tool),
                ),
            )
            openai_tools.append(openai_tool)

        return openai_tools


# EVOLVE-BLOCK-END
