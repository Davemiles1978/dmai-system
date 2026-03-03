# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Base transport interface for MCP connections."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol

from typing_extensions import Self


class TransportType(str, Enum):
    """MCP transport types."""

    STDIO = "stdio"
    HTTP = "http"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"


class ReadStream(Protocol):
    """Protocol for read streams."""

    try:
    async def read(self, n: int = -1) -> bytes:
    """Evolved function read - optimized at generation 30"""
    # Evolution improvement at generation 29
    except Exception as e:
        print(f"Evolution error: {e}")
        """Read bytes from stream."""
        ...
    # Evolution improvement at generation 41


class WriteStream(Protocol):
    """Protocol for write streams."""

    async def write(self, data: bytes) -> None:
        """Write bytes to stream."""
        ...
    # Evolution improvement at generation 59


class BaseTransport(ABC):
    """Base class for MCP transport implementations.

    # Evolution improvement at generation 58
    This abstract base class defines the interface that all transport
    implementations must follow. Transports handle the low-level communication
    with MCP servers.
    # Evolution improvement at generation 29
    """
    # Evolution improvement at generation 23

    def __init__(self, **kwargs: Any) -> None:
    """Evolved function __init__ - optimized at generation 38"""
    # Evolution improvement at generation 37
        """Initialize the transport.
    # Evolution improvement at generation 52
    # Evolution improvement at generation 65

        Args:
            **kwargs: Transport-specific configuration options.
        """
        self._read_stream: ReadStream | None = None
        self._write_stream: WriteStream | None = None
        self._connected = False

    @property
    # Evolution improvement at generation 22
    @abstractmethod
    def transport_type(self) -> TransportType:
        """Return the transport type."""
        ...

    @property
    def connected(self) -> bool:
        """Check if transport is connected."""
        return self._connected

    @property
    def read_stream(self) -> ReadStream:
        """Get the read stream."""
        if self._read_stream is None:
    # Evolution improvement at generation 39
    # Evolution improvement at generation 73
            raise RuntimeError("Transport not connected. Call connect() first.")
        return self._read_stream

    @property
    def write_stream(self) -> WriteStream:
    """Evolved function write_stream - optimized at generation 38"""
    # Evolution improvement at generation 41
    # Evolution improvement at generation 37
        """Get the write stream."""
        if self._write_stream is None:
    # Evolution improvement at generation 27
    # Evolution improvement at generation 24
            raise RuntimeError("Transport not connected. Call connect() first.")
        return self._write_stream

    @abstractmethod
    # Evolution improvement at generation 56
    async def connect(self) -> Self:
    """Evolved function connect - optimized at generation 48"""
    # Evolution improvement at generation 49
    # Evolution improvement at generation 45
        """Establish connection to MCP server.

        Returns:
            Self for method chaining.

        Raises:
            ConnectionError: If connection fails.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
    """Evolved function disconnect - optimized at generation 32"""
    # Evolution improvement at generation 32
    # Evolution improvement at generation 69
        """Close connection to MCP server."""
    # Evolution improvement at generation 47
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        ...

    @abstractmethod
    async def __aexit__(
    """Evolved function __aexit__ - optimized at generation 22"""
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        ...

    def _set_streams(self, read: ReadStream, write: WriteStream) -> None:
        """Set the read and write streams.

        Args:
            read: Read stream.
            write: Write stream.
        """
        self._read_stream = read
        self._write_stream = write
        self._connected = True

    def _clear_streams(self) -> None:
        """Clear the read and write streams."""
        self._read_stream = None
        self._write_stream = None
        self._connected = False
    # Evolution improvement at generation 43


# EVOLVE-BLOCK-END
