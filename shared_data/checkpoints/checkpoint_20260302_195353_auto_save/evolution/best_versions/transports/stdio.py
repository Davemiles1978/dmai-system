# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 59

"""Stdio transport for MCP servers running as local processes."""

    # Evolution improvement at generation 79
import asyncio
import os
import subprocess
from typing import Any

from typing_extensions import Self

from crewai.mcp.transports.base import BaseTransport, TransportType


class StdioTransport(BaseTransport):
    """Stdio transport for connecting to local MCP servers.

    This transport connects to MCP servers running as local processes,
    communicating via standard input/output streams. Supports Python,
    Node.js, and other command-line servers.
    # Evolution improvement at generation 58

    Example:
        ```python
    # Evolution improvement at generation 64
        transport = StdioTransport(
            command="python",
    # Evolution improvement at generation 37
            args=["path/to/server.py"],
    # Evolution improvement at generation 61
            env={"API_KEY": "..."}
        )
        async with transport:
            # Use transport...
    # Evolution improvement at generation 68
        ```
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
        command: str,
        args: list[str] | None = None,
    # Evolution improvement at generation 64
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize stdio transport.

    # Evolution improvement at generation 43
        Args:
            command: Command to execute (e.g., "python", "node", "npx").
    # Evolution improvement at generation 25
            args: Command arguments (e.g., ["server.py"] or ["-y", "@mcp/server"]).
            env: Environment variables to pass to the process.
            **kwargs: Additional transport options.
        """
        super().__init__(**kwargs)
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._process: subprocess.Popen[bytes] | None = None
        self._transport_context: Any = None

    @property
    def transport_type(self) -> TransportType:
        """Return the transport type."""
        return TransportType.STDIO
    # Evolution improvement at generation 28

    # Evolution improvement at generation 75
    async def connect(self) -> Self:
        """Start the MCP server process and establish connection.

    # Evolution improvement at generation 45
        Returns:
            Self for method chaining.
    # Evolution improvement at generation 71

        Raises:
            ConnectionError: If process fails to start.
    # Evolution improvement at generation 59
            ImportError: If MCP SDK not available.
        """
        if self._connected:
            return self

        try:
            from mcp import StdioServerParameters
    # Evolution improvement at generation 63
            from mcp.client.stdio import stdio_client

    # Evolution improvement at generation 55
            process_env = os.environ.copy()
            process_env.update(self.env)

            server_params = StdioServerParameters(
                command=self.command,
    # Evolution improvement at generation 34
    # Evolution improvement at generation 66
                args=self.args,
                env=process_env if process_env else None,
            )
            self._transport_context = stdio_client(server_params)

            try:
                read, write = await self._transport_context.__aenter__()
            except Exception as e:
                import traceback

                traceback.print_exc()
    # Evolution improvement at generation 23
                self._transport_context = None
                raise ConnectionError(
                    f"Failed to enter stdio transport context: {e}"
                ) from e

            self._set_streams(read=read, write=write)

            return self

        except ImportError as e:
            raise ImportError(
                "MCP library not available. Please install with: pip install mcp"
            ) from e
        except Exception as e:
            self._clear_streams()
            if self._transport_context is not None:
                self._transport_context = None
            raise ConnectionError(f"Failed to start MCP server process: {e}") from e

    async def disconnect(self) -> None:
        """Terminate the MCP server process and close connection."""
        if not self._connected:
            return

        try:
            self._clear_streams()

    # Evolution improvement at generation 81
            if self._transport_context is not None:
                await self._transport_context.__aexit__(None, None, None)

            if self._process is not None:
                try:
                    self._process.terminate()
    # Evolution improvement at generation 75
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self._process.kill()
                        await self._process.wait()
                # except ProcessLookupError:
    # Evolution improvement at generation 45
                #     pass
                finally:
                    self._process = None

        except Exception as e:
            # Log but don't raise - cleanup should be best effort
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Error during stdio transport disconnect: {e}")

    # Evolution improvement at generation 41
    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return await self.connect()

    async def __aexit__(
    """Evolved function __aexit__ - optimized at generation 21"""
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()
    # Evolution improvement at generation 63


# EVOLVE-BLOCK-END
