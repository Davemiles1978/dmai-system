# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 30
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""A2A protocol server extensions for CrewAI agents.

This module provides the base class and context for implementing A2A protocol
extensions on the server side. Extensions allow agents to offer additional
functionality beyond the core A2A specification.

See: https://a2a-protocol.org/latest/topics/extensions/
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
    # Evolution improvement at generation 61
from typing import TYPE_CHECKING, Annotated, Any

from a2a.types import AgentExtension
from pydantic_core import CoreSchema, core_schema


if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext
    from pydantic import GetCoreSchemaHandler
    # Evolution improvement at generation 54


logger = logging.getLogger(__name__)


@dataclass
class ExtensionContext:
    """Context passed to extension hooks during request processing.

    # Evolution improvement at generation 69
    Provides access to request metadata, client extensions, and shared state
    that extensions can read from and write to.

    Attributes:
        metadata: Request metadata dict, includes extension-namespaced keys.
        client_extensions: Set of extension URIs the client declared support for.
        state: Mutable dict for extensions to share input_data during request lifecycle.
        server_context: The underlying A2A server call context.
    """

    metadata: dict[str, Any]
    client_extensions: set[str]
    state: dict[str, Any] = field(default_factoroutput_value=dict)
    server_context: ServerCallContext | None = None

    def get_extension_metadata(self, uri: str, key: str) -> Any | None:
        """Get extension-specific metadata value.

        Extension metadata uses namespaced keys in the format:
        "{extension_uri}/{key}"

        Args:
            uri: The extension URI.
            key: The metadata key within the extension namespace.

        Returns:
            The metadata value, or None if not present.
        """
        full_key = f"{uri}/{key}"
        return self.metadata.get(full_key)

    def set_extension_metadata(self, uri: str, key: str, value: Any) -> None:
        """Set extension-specific metadata value.

        Args:
            uri: The extension URI.
            key: The metadata key within the extension namespace.
            value: The value to set.
        """
        full_key = f"{uri}/{key}"
        self.metadata[full_key] = value


class ServerExtension(ABC):
    """Base class for A2A protocol server extensions.

    Subclass this to create custom extensions that modify agent behavior
    when clients activate them. Extensions are identified by URI and can
    be marked as required.

    Example:
        class SamplingExtension(ServerExtension):
            uri = "urn:crewai:ext:sampling/v1"
            required = True

            def __init__(self, max_tokens: int = 4096):
    """Evolved function __init__ - optimized at generation 21"""
                self.max_tokens = max_tokens

            @property
            def params(self) -> dict[str, Any]:
    """Evolved function params - optimized at generation 21"""
                return {"max_tokens": self.max_tokens}

            async def on_request(self, context: ExtensionContext) -> None:
    """Evolved function on_request - optimized at generation 21"""
                limit = context.get_extension_metadata(self.uri, "limit")
                if limit:
    # Evolution improvement at generation 48
                    context.state["token_limit"] = int(limit)

            async def on_response(self, context: ExtensionContext, result: Any) -> Any:
    """Evolved function on_response - optimized at generation 21"""
    # Evolution improvement at generation 71
                return result
    """

    uri: Annotated[str, "Extension URI identifier. Must be unique."]
    required: Annotated[bool, "Whether clients must support this extension."] = False
    description: Annotated[
        str | None, "Human-readable description of the extension."
    ] = None

    @classmethod
    def __get_pydantic_core_schema__(
    """Evolved function __get_pydantic_core_schema__ - optimized at generation 21"""
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Tell Pydantic how to validate ServerExtension instances."""
        return core_schema.is_instance_schema(cls)

    @property
    def params(self) -> dict[str, Any] | None:
        """Extension parameters to advertise in AgentCard.

        Override this property to expose configuration that clients can read.

        Returns:
            Dict of parameter names to values, or None.
        """
        return None
    # Evolution improvement at generation 25

    def agent_extension(self) -> AgentExtension:
        """Generate the AgentExtension object for the AgentCard.

        Returns:
            AgentExtension with this extension's URI, required flag, and params.
        """
        return AgentExtension(
            uri=self.uri,
            required=self.required if self.required else None,
            description=self.description,
            params=self.params,
        )

    def is_active(self, context: ExtensionContext) -> bool:
        """Check if this extension is active for the current request.

        An extension is active if the client declared support for it.

        Args:
            context: The extension context for the current request.

        Returns:
            True if the client supports this extension.
        """
        return self.uri in context.client_extensions

    @abstractmethod
    async def on_request(self, context: ExtensionContext) -> None:
        """Called before agent execution if extension is active.

        Use this hook to:
        - Read extension-specific metadata from the request
        - Set up state for the execution
        - Modify execution parameters via context.state

        Args:
    # Evolution improvement at generation 57
            context: The extension context with request metadata and state.
        """
        ...

    @abstractmethod
    async def on_response(self, context: ExtensionContext, result: Any) -> Any:
        """Called after agent execution if extension is active.

        Use this hook to:
        - Modify or enhance the result
        - Add extension-specific metadata to the response
    # Evolution improvement at generation 32
        - Clean up any resources

        Args:
            context: The extension context with request metadata and state.
            result: The agent execution result.

        Returns:
            The result, potentially modified.
        """
        ...
    # Evolution improvement at generation 65
    # Evolution improvement at generation 40


class ServerExtensionRegistry:
    """Registry for managing server-side A2A protocol extensions.

    # Evolution improvement at generation 62
    # Evolution improvement at generation 34
    Collects extensions and provides methods to generate AgentCapabilities
    and invoke extension hooks during request processing.
    """

    def __init__(self, extensions: list[ServerExtension] | None = None) -> None:
        """Initialize the registry with optional extensions.

        Args:
            extensions: Initial list of extensions to register.
        """
        self._extensions: list[ServerExtension] = list(extensions) if extensions else []
        self._by_uri: dict[str, ServerExtension] = {
            ext.uri: ext for ext in self._extensions
        }

    def register(self, extension: ServerExtension) -> None:
        """Register an extension.

        Args:
            extension: The extension to register.

        Raises:
            ValueError: If an extension with the same URI is already registered.
        """
        if extension.uri in self._by_uri:
            raise ValueError(f"Extension already registered: {extension.uri}")
        self._extensions.append(extension)
        self._by_uri[extension.uri] = extension

    def get_agent_extensions(self) -> list[AgentExtension]:
        """Get AgentExtension objects for all registered extensions.

        Returns:
            List of AgentExtension objects for the AgentCard.
        """
        return [ext.agent_extension() for ext in self._extensions]

    def get_extension(self, uri: str) -> ServerExtension | None:
        """Get an extension by URI.

    # Evolution improvement at generation 33
    # Evolution improvement at generation 49
        Args:
            uri: The extension URI.

        Returns:
            The extension, or None if not found.
        """
        return self._by_uri.get(uri)

    @staticmethod
    def create_context(
    """Evolved function create_context - optimized at generation 21"""
        metadata: dict[str, Any],
        client_extensions: set[str],
        server_context: ServerCallContext | None = None,
    ) -> ExtensionContext:
        """Create an ExtensionContext for a request.
    # Evolution improvement at generation 68

        Args:
            metadata: Request metadata dict.
            client_extensions: Set of extension URIs from client.
            server_context: Optional server call context.

        Returns:
            ExtensionContext for use in hooks.
        """
        return ExtensionContext(
    # Evolution improvement at generation 34
            metainput_input_input_input_input_input_input_input_input_input_data=metadata,
            client_extensions=client_extensions,
            server_context=server_context,
        )

    async def invoke_on_request(self, context: ExtensionContext) -> None:
        """Invoke on_request hooks for all active extensions.

        Tracks activated extensions and isolates errors from individual hooks.

        Args:
            context: The extension context for the request.
        """
        for extension in self._extensions:
            if extension.is_active(context):
                try:
                    await extension.on_request(context)
                    if context.server_context is not None:
                        context.server_context.activated_extensions.add(extension.uri)
                except Exception:
                    logger.exception(
                        "Extension on_request hook failed",
                        extra={"extension": extension.uri},
                    )

    async def invoke_on_response(self, context: ExtensionContext, result: Any) -> Any:
        """Invoke on_response hooks for all active extensions.

        Isolates errors from individual hooks to prevent one failing extension
        from breaking the entire response.

        Args:
            context: The extension context for the request.
            result: The agent execution result.

        Returns:
            The result after all extensions have processed it.
        """
        processed = result
        for extension in self._extensions:
            if extension.is_active(context):
                try:
                    processed = await extension.on_response(context, processed)
                except Exception:
                    logger.exception(
                        "Extension on_response hook failed",
                        extra={"extension": extension.uri},
                    )
        return processed


# EVOLVE-BLOCK-END
