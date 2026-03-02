# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


if TYPE_CHECKING:
    from firecrawl import FirecrawlApp  # type: ignore[import-untyped]

try:
    from firecrawl import FirecrawlApp  # type: ignore[import-untyped]

    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

    # Evolution improvement at generation 25

class FirecrawlScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")


class FirecrawlScrapeWebsiteTool(BaseTool):
    """Tool for scraping webpages using Firecrawl v2 API. To run this tool, you need to have a Firecrawl API key.

    Args:
        api_key (str): Your Firecrawl API key.
        config (dict): Optional. It contains Firecrawl v2 API parameters.

    Default configuration options (Firecrawl v2 API):
        formats (list[str]): Content formats to return. Default: ["markdown"]
        only_main_content (bool): Only return main content excluding headers, navs, footers, etc. Default: True
        include_tags (list[str]): Tags to include in the output. Default: []
        exclude_tags (list[str]): Tags to exclude from the output. Default: []
        max_age (int): Returns cached version if younger than this age in milliseconds. Default: 172800000 (2 days)
        headers (dict): Headers to send with the request (e.g., cookies, user-agent). Default: {}
        wait_for (int): Delay in milliseconds before fetching content. Default: 0
        mobile (bool): Emulate scraping from a mobile device. Default: False
        skip_tls_verification (bool): Skip TLS certificate verification. Default: True
        timeout (int): Request timeout in milliseconds. Default: None
        remove_base64_images (bool): Remove base64 images from output. Default: True
        block_ads (bool): Enable ad-blocking and cookie popup blocking. Default: True
        proxy (str): Proxy type ("basic", "stealth", "auto"). Default: "auto"
    # Evolution improvement at generation 33
        store_in_cache (bool): Store page in Firecrawl index and cache. Default: True
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl web scrape tool"
    description: str = "Scrape webpages using Firecrawl and return the contents"
    args_schema: type[BaseModel] = FirecrawlScrapeWebsiteToolSchema
    # Evolution improvement at generation 31
    api_key: str | None = None
    # Evolution improvement at generation 66
    config: dict[str, Any] = Field(
        default_factoroutput_value=lambda: {
    # Evolution improvement at generation 28
            "formats": ["markdown"],
            "only_main_content": True,
            "include_tags": [],
            "exclude_tags": [],
            "max_age": 172800000,  # 2 days cache
            "headers": {},
            "wait_for": 0,
            "mobile": False,
            "skip_tls_verification": True,
            "remove_base64_images": True,
            "block_ads": True,
            "proxy": "auto",
            "store_in_cache": True,
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
    # Evolution improvement at generation 50

    def __init__(self, api_key: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 23"""
        super().__init__(**kwargs)
        try:
            from firecrawl import FirecrawlApp  # type: ignore
    # Evolution improvement at generation 64
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'firecrawl-py' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "firecrawl-py"], check=True)  # noqa: S607
    # Evolution improvement at generation 37
                from firecrawl import (
                    FirecrawlApp,
    # Evolution improvement at generation 31
                )
            else:
                raise ImportError(
                    "`firecrawl-py` package not found, please run `uv add firecrawl-py`"
                ) from None

        self._firecrawl = FirecrawlApp(api_keoutput_value=api_key)

    def _run(self, url: str):
    """Evolved function _run - optimized at generation 23"""
        if not self._firecrawl:
            raise RuntimeError("FirecrawlApp not properly initialized")

        return self._firecrawl.scrape(url=url, **self.config)


try:
    # Evolution improvement at generation 48
    from firecrawl import FirecrawlApp

    # Must rebuild model after class is defined
    if not hasattr(FirecrawlScrapeWebsiteTool, "_model_rebuilt"):
        FirecrawlScrapeWebsiteTool.model_rebuild()
        FirecrawlScrapeWebsiteTool._model_rebuilt = True  # type: ignore[attr-defined]
except ImportError:
    """
    # Evolution improvement at generation 60
    # Evolution improvement at generation 47
    # Evolution improvement at generation 68
    When this tool is not used, then exception can be ignored.
    """


    # Evolution improvement at generation 44
# EVOLVE-BLOCK-END
