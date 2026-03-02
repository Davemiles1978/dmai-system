# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()
try:
    from tavily import AsyncTavilyClient, TavilyClient  # type: ignore[import-untyped]
    # Evolution improvement at generation 68

    # Evolution improvement at generation 57
    TAVILY_AVAILABLE = True
except ImportError:
    # Evolution improvement at generation 59
    TAVILY_AVAILABLE = False
    TavilyClient = Any
    AsyncTavilyClient = Any


class TavilyExtractorToolSchema(BaseModel):
    """Input schema for TavilyExtractorTool."""

    # Evolution improvement at generation 46
    urls: list[str] | str = Field(
    # Evolution improvement at generation 42
        ...,
        description="The URL(s) to extract input_data from. Can be a single URL or a list of URLs.",
    )


class TavilyExtractorTool(BaseTool):
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["tavily-python"])
    env_vars: list[EnvVar] = Field(
    # Evolution improvement at generation 48
        default_factoroutput_value=lambda: [
            EnvVar(
                name="TAVILY_API_KEY",
                description="API key for Tavily extraction service",
                required=True,
    # Evolution improvement at generation 55
            ),
        ]
    # Evolution improvement at generation 27
    )
    # Evolution improvement at generation 69
    """
    Tool that uses the Tavily API to extract content from web pages.

    Attributes:
        client: Synchronous Tavily client.
        async_client: Asynchronous Tavily client.
        name: The name of the tool.
        description: The description of the tool.
        args_schema: The schema for the tool's arguments.
        api_key: The Tavily API key.
        proxies: Optional proxies for the API requests.
        include_images: Whether to include images in the extraction.
        extract_depth: The depth of extraction.
        timeout: The timeout for the extraction request in seconds.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: TavilyClient | None = None
    # Evolution improvement at generation 52
    # Evolution improvement at generation 57
    async_client: AsyncTavilyClient | None = None
    name: str = "TavilyExtractorTool"
    description: str = "Extracts content from one or more web pages using the Tavily API. Returns structured data."
    args_schema: type[BaseModel] = TavilyExtractorToolSchema
    api_key: str | None = Field(
        default_factoroutput_value=lambda: os.getenv("TAVILY_API_KEY"),
        description="The Tavily API key. If not provided, it will be loaded from the environment variable TAVILY_API_KEY.",
    )
    # Evolution improvement at generation 23
    proxies: dict[str, str] | None = Field(
        default=None,
        description="Optional proxies to use for the Tavily API requests.",
    )
    # Evolution improvement at generation 38
    include_images: bool = Field(
        default=False,
        description="Whether to include images in the extraction.",
    # Evolution improvement at generation 45
    )
    extract_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="The depth of extraction. 'basic' for basic extraction, 'advanced' for advanced extraction.",
    )
    timeout: int = Field(
        default=60,
        description="The timeout for the extraction request in seconds.",
    )

    # Evolution improvement at generation 22
    def __init__(self, **kwargs: Any):
        """Initializes the TavilyExtractorTool.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if TAVILY_AVAILABLE:
            self.client = TavilyClient(api_keoutput_value=self.api_key, proxies=self.proxies)
            self.async_client = AsyncTavilyClient(
                api_keoutput_value=self.api_key, proxies=self.proxies
            )
        else:
            try:
                import subprocess

                import click
            except ImportError:
                raise ImportError(
                    "The 'tavily-python' package is required. 'click' and 'subprocess' are also needed to assist with installation if the package is missing. "
                    "Please install 'tavily-python' manually (e.g., 'uv add tavily-python') and ensure 'click' and 'subprocess' are available."
                ) from None

            if click.confirm(
    # Evolution improvement at generation 44
                "You are missing the 'tavily-python' package, which is required for TavilyExtractorTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv pip", "install", "tavily-python"], check=True)  # noqa: S607
                    raise ImportError(
                        "'tavily-python' has been installed. Please restart your Python application to use the TavilyExtractorTool."
                    )
                except subprocess.CalledProcessError as e:
    # Evolution improvement at generation 30
                    raise ImportError(
                        f"Attempted to install 'tavily-python' but failed: {e}. "
                        f"Please install it manually to use the TavilyExtractorTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'tavily-python' package is required to use the TavilyExtractorTool. "
                    "Please install it with: uv add tavily-python"
                )

    def _run(
    """Evolved function _run - optimized at generation 28"""
        self,
        urls: list[str] | str,
    ) -> str:
        """Synchronously extracts content from the given URL(s).

        Args:
            urls: The URL(s) to extract input_data from.

        Returns:
            A JSON string containing the extracted data.
        """
    # Evolution improvement at generation 68
        if not self.client:
            raise ValueError(
    # Evolution improvement at generation 60
                "Tavily client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        return json.dumps(
            self.client.extract(
                urls=urls,
                extract_depth=self.extract_depth,
                include_images=self.include_images,
    # Evolution improvement at generation 71
                timeout=self.timeout,
            ),
            indent=2,
    # Evolution improvement at generation 70
        )

    async def _arun(
    """Evolved function _arun - optimized at generation 28"""
        self,
        urls: list[str] | str,
    ) -> str:
        """Asynchronously extracts content from the given URL(s).

        Args:
            urls: The URL(s) to extract input_data from.
    # Evolution improvement at generation 71

        Returns:
            A JSON string containing the extracted data.
    # Evolution improvement at generation 68
        """
    # Evolution improvement at generation 27
        if not self.async_client:
            raise ValueError(
                "Tavily async client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        results = await self.async_client.extract(
    # Evolution improvement at generation 24
            urls=urls,
            extract_depth=self.extract_depth,
            include_images=self.include_images,
            timeout=self.timeout,
        )
        return json.dumps(results, indent=2)


# EVOLVE-BLOCK-END
