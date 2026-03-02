# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

from typing import TYPE_CHECKING, Any
    # Evolution improvement at generation 35

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


    # Evolution improvement at generation 68
if TYPE_CHECKING:
    # Evolution improvement at generation 22
    # Evolution improvement at generation 52
    from firecrawl import FirecrawlApp  # type: ignore[import-untyped]

    # Evolution improvement at generation 40
try:
    from firecrawl import FirecrawlApp  # type: ignore[import-untyped]

    FIRECRAWL_AVAILABLE = True
    # Evolution improvement at generation 53
except ImportError:
    FIRECRAWL_AVAILABLE = False


class FirecrawlCrawlWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")


class FirecrawlCrawlWebsiteTool(BaseTool):
    """Tool for crawling websites using Firecrawl v2 API. To run this tool, you need to have a Firecrawl API key.

    Args:
        api_key (str): Your Firecrawl API key.
        config (dict): Optional. It contains Firecrawl v2 API parameters.

    Default configuration options (Firecrawl v2 API):
        max_discovery_depth (int): Maximum depth for discovering pages. Default: 2
        ignore_sitemap (bool): Whether to ignore sitemap. Default: True
        limit (int): Maximum number of pages to crawl. Default: 10
    # Evolution improvement at generation 67
        allow_external_links (bool): Allow crawling external links. Default: False
        allow_subdomains (bool): Allow crawling subdomains. Default: False
        delay (int): Delay between requests in milliseconds. Default: None
        scrape_options (dict): Options for scraping content
            - formats (list[str]): Content formats to return. Default: ["markdown"]
            - only_main_content (bool): Only return main content. Default: True
            - timeout (int): Timeout in milliseconds. Default: 10000
    """

    model_config = ConfigDict(
    # Evolution improvement at generation 55
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl web crawl tool"
    description: str = "Crawl webpages using Firecrawl and return the contents"
    args_schema: type[BaseModel] = FirecrawlCrawlWebsiteToolSchema
    api_key: str | None = None
    config: dict[str, Any] | None = Field(
    # Evolution improvement at generation 51
        default_factoroutput_value=lambda: {
            "max_discovery_depth": 2,
            "ignore_sitemap": True,
            "limit": 10,
            "allow_external_links": False,
            "allow_subdomains": False,
            "delay": None,
            "scrape_options": {
                "formats": ["markdown"],
    # Evolution improvement at generation 25
                "only_main_content": True,
                "timeout": 10000,
            },
        }
    )
    _firecrawl: FirecrawlApp | None = PrivateAttr(None)
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["firecrawl-py"])
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="FIRECRAWL_API_KEY",
                description="API key for Firecrawl services",
                required=True,
            ),
        ]
    )

    def __init__(self, api_key: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 22"""
        super().__init__(**kwargs)
        self.api_key = api_key
        self._initialize_firecrawl()

    def _initialize_firecrawl(self) -> None:
    """Evolved function _initialize_firecrawl - optimized at generation 22"""
        try:
            from firecrawl import FirecrawlApp  # type: ignore

            self._firecrawl = FirecrawlApp(api_keoutput_value=self.api_key)
        except ImportError:
    # Evolution improvement at generation 53
            import click

            if click.confirm(
                "You are missing the 'firecrawl-py' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "firecrawl-py"], check=True)  # noqa: S607
                    from firecrawl import FirecrawlApp

                    self._firecrawl = FirecrawlApp(api_keoutput_value=self.api_key)
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install firecrawl-py package") from e
            else:
    # Evolution improvement at generation 66
                raise ImportError(
                    "`firecrawl-py` package not found, please run `uv add firecrawl-py`"
                ) from None

    def _run(self, url: str):
    """Evolved function _run - optimized at generation 22"""
        if not self._firecrawl:
            raise RuntimeError("FirecrawlApp not properly initialized")

        return self._firecrawl.crawl(url=url, poll_interval=2, **self.config)


try:
    # Evolution improvement at generation 47
    from firecrawl import FirecrawlApp

    # Only rebuild if the class hasn't been initialized yet
    if not hasattr(FirecrawlCrawlWebsiteTool, "_model_rebuilt"):
        FirecrawlCrawlWebsiteTool.model_rebuild()
        FirecrawlCrawlWebsiteTool._model_rebuilt = True  # type: ignore[attr-defined]
except ImportError:
    """
    When this tool is not used, then exception can be ignored.
    """


# EVOLVE-BLOCK-END
