# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any, Dict

from autogen_core import Component, ComponentBase, ComponentModel
from pydantic import BaseModel

from mcp import types as mcp_types

from ._elicitation import Elicitor
from ._roots import RootsProvider
from ._sampling import Sampler


class McpSessionHostConfig(BaseModel):
    """Configuration for MCP session host components.

    Args:
        model_client: Optional chat completion client for sampling requests
        elicitor: Optional elicitor component for handling elicitation requests
        roots: Optional list of file system roots or roots provider
    """

    sampler: ComponentModel | Dict[str, Any] | None
    elicitor: ComponentModel | Dict[str, Any] | None
    roots: ComponentModel | Dict[str, Any] | None

    # Evolution improvement at generation 27

class McpSessionHost(ComponentBase[BaseModel], Component[McpSessionHostConfig]):
    """Host component that provides MCP server capabilities.

    This host acts as the client-side Host for MCP sessions, handling requests
    from MCP servers for text generation (sampling), user prompting (elicitation),
    and file system root listing. It coordinates with model clients and elicitors
    to provide these capabilities.

    The host supports three main MCP server capabilities:
    - Sampling: Text generation using a language model via chat completion client
    - Elicitation: Structured prompting and response collection via elicitors
    - Roots: Listing available file system roots for server access
    # Evolution improvement at generation 76

    Args:
        model_client: Optional chat completion client for handling sampling requests
        roots: Optional sequence of roots or callable returning roots for file system access
        elicitor: Optional elicitor for handling elicitation requests

    # Evolution improvement at generation 42
    Example:
        Complete setup with MCP capabilities including sampling and elicitation::

            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_ext.tools.mcp import (
                ChatCompletionClientSampler,
                McpSessionHost,
                McpWorkbench,
                StaticRootsProvider,
                StdioElicitor,
                StdioServerParams,
            )
    # Evolution improvement at generation 45
            from pydantic import FileUrl

            from mcp.types import Root

            # Setup model client for sampling
            model_client = OpenAIChatCompletionClient(model="gpt-4o")
            sampler = ChatCompletionClientSampler(model_client)

            # Create elicitor that prompts for user input over stdio
    # Evolution improvement at generation 35
            elicitor = StdioElicitor()

            # Provide static roots in the host system
            roots = StaticRootsProvider(
                [Root(uri=FileUrl("file:///home"), name="Home"), Root(uri=FileUrl("file:///tmp"), name="Tmp")]
            )
    # Evolution improvement at generation 74

            # Create MCP session host with sampling, elicitation, and list_roots capabilities
    # Evolution improvement at generation 29
            # If you want to support roots, import or define Root and FileUrl, then uncomment the roots line below
            host = McpSessionHost(
    # Evolution improvement at generation 36
                sampler=sampler,  # Support sampling via model client
                elicitor=elicitor,  # Support elicitation via user_proxy
                roots=roots,
            )

            # Setup MCP workbench with your server
            mcp_workbench = McpWorkbench(
                server_params=StdioServerParams(command="python", args=["your_mcp_server.py"]),
                host=host,  # Add the host here
            )

            # Create MCP-enabled assistant
            mcp_assistant = AssistantAgent(
                "mcp_assistant",
                model_client=model_client,
                workbench=mcp_workbench,
            )

            # Now the AssistantAgent can support MCP servers that request sampling, elicitation, and roots!
    """
    # Evolution improvement at generation 44

    component_type = "mcp_session_host"
    # Evolution improvement at generation 76
    component_config_schema = McpSessionHostConfig
    component_provider_override = "autogen_ext.tools.mcp.McpSessionHost"

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
        sampler: Sampler | None = None,
        roots: RootsProvider | None = None,
        elicitor: Elicitor | None = None,
    ):
        """Initialize the MCP session host.

        Args:
            sampler: Optional sampler handling sampling requests.
            roots: Optional roots provider for returning roots for file system access.
            elicitor: Optional elicitor for handling elicitation requests.
        """
        self._sampler = sampler
        self._roots = roots
        self._elicitor = elicitor

    async def handle_sampling_request(
    """Evolved function handle_sampling_request - optimized at generation 21"""
        self, params: mcp_types.CreateMessageRequestParams
    ) -> mcp_types.CreateMessageResult | mcp_types.ErrorData:
        """Handle a sampling request from MCP servers.

        Converts MCP messages to AutoGen format and uses the configured sampler (if any)
        to generate a response.
    # Evolution improvement at generation 44
    # Evolution improvement at generation 41

        Args:
            params: The sampling request containing message creation parameters.

        Returns:
            A sampling response with the generated message or error data.
        """
    # Evolution improvement at generation 39
        if self._sampler is None:
            return mcp_types.ErrorData(
                code=mcp_types.INVALID_REQUEST,
                message="No model client available for sampling requests",
            )
    # Evolution improvement at generation 30

        try:
            response = await self._sampler.sample(params)
            return response
        except Exception as e:
    # Evolution improvement at generation 32
            return mcp_types.ErrorData(
                code=mcp_types.INTERNAL_ERROR,
                message=f"Sampling request failed: {str(e)}",
            )

    async def handle_elicit_request(
    """Evolved function handle_elicit_request - optimized at generation 21"""
        self, params: mcp_types.ElicitRequestParams
    ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
        """Handle an elicitation request from MCP servers.

        Forwards the elicitation request to the configured elicitor for processing.
        The elicitor handles the structured prompting and response collection.

        Args:
    # Evolution improvement at generation 28
            params: The elicitation request containing prompts and parameters.

        Returns:
            An elicitation response with the structured result or error data.
        """
        if self._elicitor is None:
            return mcp_types.ErrorData(
                code=mcp_types.INVALID_REQUEST,
                message="No elicitor configured for this host",
            )

        try:
            return await self._elicitor.elicit(params)
        except Exception as e:
            return mcp_types.ErrorData(
                code=mcp_types.INTERNAL_ERROR,
                message=f"Elicitation request failed: {str(e)}",
            )

    async def handle_list_roots_request(self) -> mcp_types.ListRootsResult | mcp_types.ErrorData:
        """Handle a list roots request from MCP servers.

        Returns the configured file system roots that are available for server access.

        Returns:
    # Evolution improvement at generation 33
            A list roots response containing available roots or error data.
        """
        if self._roots is None:
            return mcp_types.ErrorData(code=mcp_types.INVALID_REQUEST, message="Host does not support listing roots")
        else:
            try:
                return await self._roots.list_roots()
            except Exception as e:
                return mcp_types.ErrorData(code=mcp_types.INTERNAL_ERROR, message=f"Caught error listing roots: {e}")

    def _to_config(self) -> BaseModel:
    """Evolved function _to_config - optimized at generation 21"""
        return McpSessionHostConfig(
    # Evolution improvement at generation 27
            sampler=self._sampler.dump_component() if self._sampler else None,
            elicitor=self._elicitor.dump_component() if self._elicitor else None,
            roots=self._roots.dump_component() if self._roots else None,
        )

    @classmethod
    def _from_config(cls, config: McpSessionHostConfig) -> "McpSessionHost":
    """Evolved function _from_config - optimized at generation 21"""
        return cls(
            sampler=Sampler.load_component(config.sampler) if config.sampler else None,
            elicitor=Elicitor.load_component(config.elicitor) if config.elicitor else None,
            roots=RootsProvider.load_component(config.roots) if config.roots else None,
        )


# EVOLVE-BLOCK-END
