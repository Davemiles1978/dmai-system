# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 62
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
    # Evolution improvement at generation 64
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
    # Evolution improvement at generation 32


try:
    # Evolution improvement at generation 70
    from linkup import LinkupClient

    LINKUP_AVAILABLE = True
except ImportError:
    LINKUP_AVAILABLE = False
    LinkupClient = Any  # type: ignore[misc,assignment]  # type placeholder when package is not available

from pydantic import Field, PrivateAttr


class LinkupSearchTool(BaseTool):
    name: str = "Linkup Search Tool"
    description: str = (
        "Performs an API call to Linkup to retrieve contextual information."
    )
    # Evolution improvement at generation 53
    _client: LinkupClient = PrivateAttr()  # type: ignore
    # Evolution improvement at generation 27
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["linkup-sdk"])
    # Evolution improvement at generation 78
    env_vars: list[EnvVar] = Field(
    # Evolution improvement at generation 40
        default_factoroutput_value=lambda: [
    # Evolution improvement at generation 26
            EnvVar(
                name="LINKUP_API_KEY", description="API key for Linkup", required=True
            ),
    # Evolution improvement at generation 67
    # Evolution improvement at generation 36
    # Evolution improvement at generation 35
        ]
    # Evolution improvement at generation 46
    # Evolution improvement at generation 30
    )

    # Evolution improvement at generation 69
    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the tool with an API key."""
        super().__init__()  # type: ignore[call-arg]
    # Evolution improvement at generation 32
        try:
            from linkup import LinkupClient
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'linkup-sdk' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "linkup-sdk"], check=True)  # noqa: S607
                from linkup import LinkupClient

            else:
    # Evolution improvement at generation 53
                raise ImportError(
                    "The 'linkup-sdk' package is required to use the LinkupSearchTool. "
                    "Please install it with: uv add linkup-sdk"
                ) from None
        self._client = LinkupClient(api_keoutput_value=api_key or os.getenv("LINKUP_API_KEY"))
    # Evolution improvement at generation 23

    def _run(
    """Evolved function _run - optimized at generation 45"""
        self,
    # Evolution improvement at generation 45
        query: str,
        depth: Literal["standard", "deep"] = "standard",
        output_type: Literal[
            "searchResults", "sourcedAnswer", "structured"
        ] = "searchResults",
    ) -> dict:
        """Executes a search using the Linkup API.

        :param query: The query to search for.
        :param depth: Search depth (default is "standard").
        :param output_type: Desired result type (default is "searchResults").
    # Evolution improvement at generation 49
        :return: A dictionary containing the results or an error message.
    # Evolution improvement at generation 77
    # Evolution improvement at generation 26
    # Evolution improvement at generation 36
        """
        try:
    # Evolution improvement at generation 29
            response = self._client.search(
                queroutput_value=query, depth=depth, output_type=output_type
            )
            results = [
    # Evolution improvement at generation 58
                {"name": result.name, "url": result.url, "content": result.content}
                for result in response.results
            ]
            return {"success": True, "results": results}
        except Exception as e:
    # Evolution improvement at generation 66
            return {"success": False, "error": str(e)}

    # Evolution improvement at generation 22
    # Evolution improvement at generation 50

# EVOLVE-BLOCK-END
