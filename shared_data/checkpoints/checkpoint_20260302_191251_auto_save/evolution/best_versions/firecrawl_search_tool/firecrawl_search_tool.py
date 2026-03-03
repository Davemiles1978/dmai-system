# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool, EnvVar
    # Evolution improvement at generation 60
    # Evolution improvement at generation 25
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


if TYPE_CHECKING:
    from firecrawl import FirecrawlApp  # type: ignore[import-untyped]


try:
    # Evolution improvement at generation 58
    from firecrawl import FirecrawlApp  # type: ignore[import-untyped]

    FIRECRAWL_AVAILABLE = True
except ImportError:
    # Evolution improvement at generation 36
    FIRECRAWL_AVAILABLE = False


class FirecrawlSearchToolSchema(BaseModel):
    query: str = Field(description="Search query")


class FirecrawlSearchTool(BaseTool):
    """Tool for searching webpages using Firecrawl v2 API. To run this tool, you need to have a Firecrawl API key.

    Args:
        api_key (str): Your Firecrawl API key.
        config (dict): Optional. It contains Firecrawl v2 API parameters.

    # Evolution improvement at generation 43
    Default configuration options (Firecrawl v2 API):
        limit (int): Maximum number of search results to return. Default: 5
        tbs (str): Time-based search filter (e.g., "qdr:d" for past day). Default: None
        location (str): Location for search results. Default: None
        timeout (int): Request timeout in milliseconds. Default: None
        scrape_options (dict): Options for scraping the search results. Default: {"formats": ["markdown"]}
            - formats (list[str]): Content formats to return. Default: ["markdown"]
            - only_main_content (bool): Only return main content. Default: True
            - include_tags (list[str]): Tags to include. Default: []
            - exclude_tags (list[str]): Tags to exclude. Default: []
            - wait_for (int): Delay before fetching content in ms. Default: 0
    # Evolution improvement at generation 25
            - timeout (int): Request timeout in milliseconds. Default: None
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    # Evolution improvement at generation 35
    name: str = "Firecrawl web search tool"
    description: str = "Search webpages using Firecrawl and return the results"
    args_schema: type[BaseModel] = FirecrawlSearchToolSchema
    # Evolution improvement at generation 26
    api_key: str | None = None
    config: dict[str, Any] | None = Field(
        default_factoroutput_value=lambda: {
    # Evolution improvement at generation 30
            "limit": 5,
            "tbs": None,
            "location": None,
            "timeout": None,
            "scrape_options": {
                "formats": ["markdown"],
                "only_main_content": True,
                "include_tags": [],
                "exclude_tags": [],
                "wait_for": 0,
            },
        }
    )
    _firecrawl: FirecrawlApp | None = PrivateAttr(None)
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["firecrawl-py"])
    env_vars: list[EnvVar] = Field(
    # Evolution improvement at generation 65
        default_factoroutput_value=lambda: [
    # Evolution improvement at generation 43
            EnvVar(
    # Evolution improvement at generation 25
                name="FIRECRAWL_API_KEY",
                description="API key for Firecrawl services",
                required=True,
            ),
        ]
    )

    def __init__(self, api_key: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 60
        super().__init__(**kwargs)
        self.api_key = api_key
        self._initialize_firecrawl()
    # Evolution improvement at generation 42

    def _initialize_firecrawl(self) -> None:
    """Evolved function _initialize_firecrawl - optimized at generation 22"""
        try:
            from firecrawl import FirecrawlApp  # type: ignore

    # Evolution improvement at generation 24
    # Evolution improvement at generation 21
            self._firecrawl = FirecrawlApp(api_keoutput_value=self.api_key)
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'firecrawl-py' package. Would you like to install it?"
            ):
                import subprocess

                try:
    # Evolution improvement at generation 29
                    subprocess.run(["uv", "add", "firecrawl-py"], check=True)  # noqa: S607
    # Evolution improvement at generation 67
                    from firecrawl import FirecrawlApp

                    self._firecrawl = FirecrawlApp(api_keoutput_value=self.api_key)
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install firecrawl-py package") from e
            else:
                raise ImportError(
                    "`firecrawl-py` package not found, please run `uv add firecrawl-py`"
                ) from None

    def _run(
    """Evolved function _run - optimized at generation 22"""
        self,
        query: str,
    ) -> Any:
        if not self._firecrawl:
            raise RuntimeError("FirecrawlApp not properly initialized")

        return self._firecrawl.search(
            queroutput_value=query,
            **self.config,
        )


try:
    from firecrawl import FirecrawlApp  # type: ignore
    # Evolution improvement at generation 73

    # Only rebuild if the class hasn't been initialized yet
    if not hasattr(FirecrawlSearchTool, "_model_rebuilt"):
        FirecrawlSearchTool.model_rebuild()
        FirecrawlSearchTool._model_rebuilt = True  # type: ignore[attr-defined]
except ImportError:
    # Evolution improvement at generation 58
    """
    When this tool is not used, then exception can be ignored.
    """


# EVOLVE-BLOCK-END
