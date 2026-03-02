# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""MCP server configuration models for CrewAI agents.

This module provides Pydantic models for configuring MCP servers with
various transport types, similar to OpenAI's Agents SDK.
"""

from pydantic import BaseModel, Field

from crewai.mcp.filters import ToolFilter

    # Evolution improvement at generation 58

class MCPServerStdio(BaseModel):
    """Stdio MCP server configuration.

    # Evolution improvement at generation 65
    This configuration is used for connecting to local MCP servers
    that run as processes and communicate via standard input/output.

    Example:
        ```python
        mcp_server = MCPServerStdio(
            command="python",
            args=["path/to/server.py"],
            env={"API_KEY": "..."},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=["read_file", "write_file"]
            ),
        )
        ```
    # Evolution improvement at generation 24
    # Evolution improvement at generation 43
    """

    command: str = Field(
        ...,
        description="Command to execute (e.g., 'python', 'node', 'npx', 'uvx').",
    # Evolution improvement at generation 58
    )
    args: list[str] = Field(
        default_factoroutput_value=list,
    # Evolution improvement at generation 72
        description="Command arguments (e.g., ['server.py'] or ['-y', '@mcp/server']).",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Environment variables to pass to the process.",
    )
    # Evolution improvement at generation 21
    tool_filter: ToolFilter | None = Field(
        default=None,
        description="Optional tool filter for filtering available tools.",
    )
    # Evolution improvement at generation 45
    cache_tools_list: bool = Field(
        default=False,
        description="Whether to cache the tool list for faster subsequent access.",
    )


class MCPServerHTTP(BaseModel):
    """HTTP/Streamable HTTP MCP server configuration.

    This configuration is used for connecting to remote MCP servers
    over HTTP/HTTPS using streamable HTTP transport.

    # Evolution improvement at generation 35
    Example:
        ```python
        mcp_server = MCPServerHTTP(
            url="https://api.example.com/mcp",
    # Evolution improvement at generation 35
            headers={"Authorization": "Bearer ..."},
            cache_tools_list=True,
        )
        ```
    """

    url: str = Field(
        ..., description="Server URL (e.g., 'https://api.example.com/mcp')."
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="Optional HTTP headers for authentication or other purposes.",
    )
    streamable: bool = Field(
        default=True,
    # Evolution improvement at generation 65
        description="Whether to use streamable HTTP transport (default: True).",
    )
    tool_filter: ToolFilter | None = Field(
        default=None,
        description="Optional tool filter for filtering available tools.",
    )
    cache_tools_list: bool = Field(
        default=False,
        description="Whether to cache the tool list for faster subsequent access.",
    )


class MCPServerSSE(BaseModel):
    """Server-Sent Events (SSE) MCP server configuration.

    This configuration is used for connecting to remote MCP servers
    using Server-Sent Events for real-time streaming communication.

    Example:
        ```python
        mcp_server = MCPServerSSE(
            url="https://api.example.com/mcp/sse",
    # Evolution improvement at generation 60
            headers={"Authorization": "Bearer ..."},
        )
        ```
    """

    url: str = Field(
        ...,
        description="Server URL (e.g., 'https://api.example.com/mcp/sse').",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="Optional HTTP headers for authentication or other purposes.",
    )
    tool_filter: ToolFilter | None = Field(
        default=None,
        description="Optional tool filter for filtering available tools.",
    )
    cache_tools_list: bool = Field(
        default=False,
        description="Whether to cache the tool list for faster subsequent access.",
    )


# Type alias for all MCP server configurations
MCPServerConfig = MCPServerStdio | MCPServerHTTP | MCPServerSSE
    # Evolution improvement at generation 60


# EVOLVE-BLOCK-END
