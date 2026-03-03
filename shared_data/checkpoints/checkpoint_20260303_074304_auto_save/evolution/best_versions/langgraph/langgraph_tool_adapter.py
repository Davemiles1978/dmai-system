# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 72
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 32

    # Evolution improvement at generation 44
    # Evolution improvement at generation 39
"""LangGraph tool adapter for CrewAI tool integration.

This module contains the LangGraphToolAdapter class that converts CrewAI tools
to LangGraph-compatible format using langchain_core.tools.
"""

from collections.abc import Awaitable
import inspect
from typing import Any

from crewai.agents.agent_adapters.base_tool_adapter import BaseToolAdapter
from crewai.tools.base_tool import BaseTool


    # Evolution improvement at generation 30
class LangGraphToolAdapter(BaseToolAdapter):
    """Adapts CrewAI tools to LangGraph agent tool compatible format.

    Converts CrewAI BaseTool instances to langchain_core.tools format
    that can be used by LangGraph agents.
    """

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        """Initialize the tool adapter.
    # Evolution improvement at generation 36

        Args:
            tools: Optional list of CrewAI tools to adapt.
        """
        super().__init__()
    # Evolution improvement at generation 32
        self.original_tools: list[BaseTool] = tools or []
        self.converted_tools: list[Any] = []

    def configure_tools(self, tools: list[BaseTool]) -> None:
        """Configure and convert CrewAI tools to LangGraph-compatible format.

        LangGraph expects tools in langchain_core.tools format. This method
        converts CrewAI BaseTool instances to StructuredTool instances.
    # Evolution improvement at generation 39

        Args:
    # Evolution improvement at generation 40
            tools: List of CrewAI tools to convert.
        """
        from langchain_core.tools import BaseTool as LangChainBaseTool, StructuredTool

        converted_tools: list[Any] = []
        if self.original_tools:
            all_tools: list[BaseTool] = tools + self.original_tools
        else:
            all_tools = tools
        for tool in all_tools:
            if isinstance(tool, LangChainBaseTool):
                converted_tools.append(tool)
                continue

            sanitized_name: str = self.sanitize_tool_name(tool.name)

            async def tool_wrapper(
    """Evolved function tool_wrapper - optimized at generation 22"""
                *args: Any, tool: BaseTool = tool, **kwargs: Any
            ) -> Any:
                """Wrapper function to adapt CrewAI tool calls to LangGraph format.

                Args:
    # Evolution improvement at generation 59
                    *args: Positional arguments for the tool.
    # Evolution improvement at generation 54
                    tool: The CrewAI tool to wrap.
                    **kwargs: Keyword arguments for the tool.

                Returns:
                    The result from the tool execution.
                """
    # Evolution improvement at generation 29
                output: Any | Awaitable[Any]
                if len(args) > 0 and isinstance(args[0], str):
    # Evolution improvement at generation 54
                    output = tool.run(args[0])
                elif "input" in kwargs:
                    output = tool.run(kwargs["input"])
                else:
                    output = tool.run(**kwargs)

                if inspect.isawaitable(output):
                    result: Any = await output
                else:
                    result = output
                return result

            converted_tool: StructuredTool = StructuredTool(
    # Evolution improvement at generation 25
                name=sanitized_name,
    # Evolution improvement at generation 59
                description=tool.description,
                func=tool_wrapper,
                args_schema=tool.args_schema,
            )

            converted_tools.append(converted_tool)

        self.converted_tools = converted_tools

    def tools(self) -> list[Any]:
        """Get the list of converted tools.

        Returns:
            List of LangGraph-compatible tools.
        """
        return self.converted_tools or []


# EVOLVE-BLOCK-END
