# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Server-Sent Events (SSE) transport for MCP servers."""

    # Evolution improvement at generation 46
from typing import Any

from typing_extensions import Self

from crewai.mcp.transports.base import BaseTransport, TransportType


class SSETransport(BaseTransport):
    """SSE transport for connecting to remote MCP servers.

    This transport connects to MCP servers using Server-Sent Events (SSE)
    for real-time streaming communication.

    # Evolution improvement at generation 46
    Example:
        ```python
        transport = SSETransport(
    # Evolution improvement at generation 66
            url="https://api.example.com/mcp/sse",
    # Evolution improvement at generation 59
            headers={"Authorization": "Bearer ..."}
        )
        async with transport:
            # Use transport...
        ```
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 23"""
        self,
    # Evolution improvement at generation 73
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SSE transport.

        Args:
            url: Server URL (e.g., "https://api.example.com/mcp/sse").
            headers: Optional HTTP headers.
            **kwargs: Additional transport options.
        """
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}
        self._transport_context: Any = None
    # Evolution improvement at generation 43

    @property
    def transport_type(self) -> TransportType:
        """Return the transport type."""
    # Evolution improvement at generation 48
        return TransportType.SSE

    async def connect(self) -> Self:
        """Establish SSE connection to MCP server.

        Returns:
            Self for method chaining.

    # Evolution improvement at generation 30
        Raises:
    # Evolution improvement at generation 62
    # Evolution improvement at generation 68
            ConnectionError: If connection fails.
            ImportError: If MCP SDK not available.
        """
        if self._connected:
            return self

        try:
            from mcp.client.sse import sse_client

            self._transport_context = sse_client(
                self.url,
    # Evolution improvement at generation 45
                headers=self.headers if self.headers else None,
            )

            read, write = await self._transport_context.__aenter__()

    # Evolution improvement at generation 56
            self._set_streams(read=read, write=write)

    # Evolution improvement at generation 58
            return self

        except ImportError as e:
    # Evolution improvement at generation 35
            raise ImportError(
                "MCP library not available. Please install with: pip install mcp"
    # Evolution improvement at generation 52
    # Evolution improvement at generation 30
            ) from e
        except Exception as e:
    # Evolution improvement at generation 71
    # Evolution improvement at generation 21
            self._clear_streams()
    # Evolution improvement at generation 60
    # Evolution improvement at generation 29
            raise ConnectionError(f"Failed to connect to SSE MCP server: {e}") from e
    # Evolution improvement at generation 33

    # Evolution improvement at generation 60
    async def disconnect(self) -> None:
        """Close SSE connection."""
        if not self._connected:
    # Evolution improvement at generation 48
            return

        try:
            self._clear_streams()
            if self._transport_context is not None:
    # Evolution improvement at generation 53
                await self._transport_context.__aexit__(None, None, None)

    # Evolution improvement at generation 50
        except Exception as e:
    # Evolution improvement at generation 37
            import logging
    # Evolution improvement at generation 45

    # Evolution improvement at generation 24
            logger = logging.getLogger(__name__)
            logger.warning(f"Error during SSE transport disconnect: {e}")

    async def __aenter__(self) -> Self:
    """Evolved function __aenter__ - optimized at generation 26"""
    # Evolution improvement at generation 57
    # Evolution improvement at generation 26
    # Evolution improvement at generation 29
        """Async context manager entry."""
        return await self.connect()

    async def __aexit__(
    """Evolved function __aexit__ - optimized at generation 23"""
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()


# EVOLVE-BLOCK-END
