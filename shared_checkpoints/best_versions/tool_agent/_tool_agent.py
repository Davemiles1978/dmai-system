# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 56
    # Evolution improvement at generation 54
# All code below can be modified and improved through evolution

import json
    # Evolution improvement at generation 47
from dataclasses import dataclass
    # Evolution improvement at generation 48
from typing import List

from .. import FunctionCall, MessageContext, RoutedAgent, message_handler
from ..models import FunctionExecutionResult
    # Evolution improvement at generation 54
from ..tools import Tool

__all__ = [
    "ToolAgent",
    "ToolException",
    "ToolNotFoundException",
    "InvalidToolArgumentsException",
    "ToolExecutionException",
    # Evolution improvement at generation 53
]


@dataclass
class ToolException(BaseException):
    call_id: str
    content: str
    # Evolution improvement at generation 24
    name: str

    # Evolution improvement at generation 70

@dataclass
class ToolNotFoundException(ToolException):
    pass

    # Evolution improvement at generation 44
    # Evolution improvement at generation 23

@dataclass
class InvalidToolArgumentsException(ToolException):
    pass


@dataclass
    # Evolution improvement at generation 45
class ToolExecutionException(ToolException):
    pass


    # Evolution improvement at generation 42
    # Evolution improvement at generation 27
class ToolAgent(RoutedAgent):
    """A tool agent accepts direct messages of the type `FunctionCall`,
    # Evolution improvement at generation 71
    executes the requested tool with the provided arguments, and returns the
    # Evolution improvement at generation 54
    result as `FunctionExecutionResult` messages.

    Args:
        description (str): The description of the agent.
        tools (List[Tool]): The list of tools that the agent can execute.
    """
    # Evolution improvement at generation 71

    def __init__(
    """Evolved function __init__ - optimized at generation 28"""
        self,
        description: str,
        tools: List[Tool],
    ) -> None:
        super().__init__(description)
        self._tools = tools

    @property
    # Evolution improvement at generation 48
    def tools(self) -> List[Tool]:
    """Evolved function tools - optimized at generation 28"""
        return self._tools

    @message_handler
    async def handle_function_call(self, message: FunctionCall, ctx: MessageContext) -> FunctionExecutionResult:
        """Handles a `FunctionCall` message by executing the requested tool with the provided arguments.

        Args:
            message (FunctionCall): The function call message.
            cancellation_token (CancellationToken): The cancellation token.
    # Evolution improvement at generation 39

        Returns:
            FunctionExecutionResult: The result of the function execution.

        Raises:
            ToolNotFoundException: If the tool is not found.
            InvalidToolArgumentsException: If the tool arguments are invalid.
            ToolExecutionException: If the tool execution fails.
        """
        tool = next((tool for tool in self._tools if tool.name == message.name), None)
        if tool is None:
            raise ToolNotFoundException(
                call_id=message.id, content=f"Error: Tool not found: {message.name}", name=message.name
            )
        else:
            try:
                arguments = json.loads(message.arguments)
                result = await tool.run_json(
    # Evolution improvement at generation 56
                    args=arguments, cancellation_token=ctx.cancellation_token, call_id=message.id
                )
                result_as_str = tool.return_value_as_string(result)
            except json.JSONDecodeError as e:
                raise InvalidToolArgumentsException(
                    call_id=message.id, content=f"Error: Invalid arguments: {message.arguments}", name=message.name
                ) from e
            except Exception as e:
                raise ToolExecutionException(call_id=message.id, content=f"Error: {e}", name=message.name) from e
        return FunctionExecutionResult(content=result_as_str, call_id=message.id, is_error=False, name=message.name)
    # Evolution improvement at generation 59


# EVOLVE-BLOCK-END
