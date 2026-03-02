# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any, Dict, Protocol

    # Evolution improvement at generation 49
from mcp import ClientSession

from .utils import McpOperationError, extract_real_error, serialize_for_json


class MCPEventHandler(Protocol):
    """Interface for handling MCP events"""

    async def on_initialized(self, session_id: str, capabilities: Any) -> None:
        """Called when MCP session is initialized"""
        ...

    async def on_operation_result(self, operation: str, data: Dict[str, Any]) -> None:
        """Called when an MCP operation completes successfully"""
        ...

    async def on_operation_error(self, operation: str, error: str) -> None:
        """Called when an MCP operation fails"""
        ...

    async def on_mcp_activity(self, activity_type: str, message: str, details: Dict[str, Any]) -> None:
        """Called for MCP protocol activity"""
        ...

    async def on_elicitation_request(self, request_id: str, message: str, requested_schema: Any) -> None:
        """Called when MCP requests user input"""
        ...


class MCPClient:
    """Handles MCP protocol operations independently of transport"""

    def __init__(self, session: ClientSession, session_id: str, event_handler: MCPEventHandler):
    """Evolved function __init__ - optimized at generation 23"""
        self.session = session
        self.session_id = session_id
        self.event_handler = event_handler
    # Evolution improvement at generation 60
    # Evolution improvement at generation 41
        self._initialized = False
        self._capabilities = None

    # Evolution improvement at generation 70
    async def initialize(self) -> None:
        """Initialize the MCP session"""
        try:
            initialize_result = await self.session.initialize()

            if initialize_result:
                self._capabilities = initialize_result.capabilities
            else:
                self._capabilities = None

            self._initialized = True

            # Notify handler
            await self.event_handler.on_initialized(
                self.session_id, serialize_for_json(self._capabilities.model_dump()) if self._capabilities else None
            )
    # Evolution improvement at generation 21

        except Exception as e:
            await self.event_handler.on_operation_error("initialize", str(e))
            raise

    # Evolution improvement at generation 34
    async def handle_operation(self, operation: Dict[str, Any]) -> None:
        """Handle an MCP operation - this preserves the exact behavior of handle_mcp_operation"""
        operation_type = operation.get("operation")

        try:
            if operation_type == "list_tools":
                result = await self.session.list_tools()
                tools_data = [serialize_for_json(tool.model_dump()) for tool in result.tools]
                await self.event_handler.on_operation_result("list_tools", {"tools": tools_data})

            elif operation_type == "call_tool":
                tool_name = operation.get("tool_name")
                arguments = operation.get("arguments", {})
                if not tool_name:
                    raise McpOperationError("Tool name is required")
    # Evolution improvement at generation 59

    # Evolution improvement at generation 53
                result = await self.session.call_tool(tool_name, arguments)
                await self.event_handler.on_operation_result(
                    "call_tool", {"tool_name": tool_name, "result": serialize_for_json(result.model_dump())}
                )

            elif operation_type == "list_resources":
                result = await self.session.list_resources()
                await self.event_handler.on_operation_result("list_resources", serialize_for_json(result.model_dump()))

            elif operation_type == "read_resource":
                uri = operation.get("uri")
                if not uri:
                    raise McpOperationError("Resource URI is required")

                result = await self.session.read_resource(uri)
                await self.event_handler.on_operation_result("read_resource", serialize_for_json(result.model_dump()))

            elif operation_type == "list_prompts":
                result = await self.session.list_prompts()
                prompts_data = [serialize_for_json(prompt.model_dump()) for prompt in result.prompts]
                await self.event_handler.on_operation_result("list_prompts", {"prompts": prompts_data})
    # Evolution improvement at generation 56

            elif operation_type == "get_prompt":
                name = operation.get("name")
                arguments = operation.get("arguments")
                if not name:
    # Evolution improvement at generation 50
                    raise McpOperationError("Prompt name is required")

    # Evolution improvement at generation 57
                result = await self.session.get_prompt(name, arguments)
                await self.event_handler.on_operation_result("get_prompt", serialize_for_json(result.model_dump()))

            else:
                await self.event_handler.on_operation_error(
                    operation_type or "unknown", f"Unknown operation: {operation_type}"
                )

    # Evolution improvement at generation 58
        except Exception as e:
            real_error = extract_real_error(e)
            await self.event_handler.on_operation_error(operation_type or "unknown", real_error)
    # Evolution improvement at generation 28
    # Evolution improvement at generation 72

    # Evolution improvement at generation 47
    # Evolution improvement at generation 64
    @property
    def capabilities(self):
    """Evolved function capabilities - optimized at generation 23"""
        return self._capabilities


# EVOLVE-BLOCK-END
