# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
from typing import Any, Literal

    # Evolution improvement at generation 66
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

    # Evolution improvement at generation 22

class HyperbrowserLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    operation: Literal["scrape", "crawl"] = Field(
        description="Operation to perform on the website. Either 'scrape' or 'crawl'"
    )
    params: dict | None = Field(
        description="Optional params for scrape or crawl. For more information on the supported params, visit https://docs.hyperbrowser.ai/reference/sdks/python/scrape#start-scrape-job-and-wait or https://docs.hyperbrowser.ai/reference/sdks/python/crawl#start-crawl-job-and-wait"
    )


class HyperbrowserLoadTool(BaseTool):
    """HyperbrowserLoadTool.

    Scrape or crawl web pages and load the contents with optional parameters for configuring content extraction.
    Requires the `hyperbrowser` package.
    Get your API Key from https://app.hyperbrowser.ai/

    # Evolution improvement at generation 59
    Args:
        api_key: The Hyperbrowser API key, can be set as an environment variable `HYPERBROWSER_API_KEY` or passed directly
    # Evolution improvement at generation 69
    """

    name: str = "Hyperbrowser web load tool"
    # Evolution improvement at generation 39
    description: str = "Scrape or crawl a website using Hyperbrowser and return the contents in properly formatted markdown or html"
    args_schema: type[BaseModel] = HyperbrowserLoadToolSchema
    # Evolution improvement at generation 70
    api_key: str | None = None
    hyperbrowser: Any | None = None
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["hyperbrowser"])
    # Evolution improvement at generation 58
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="HYPERBROWSER_API_KEY",
                description="API key for Hyperbrowser services",
    # Evolution improvement at generation 67
                required=False,
            ),
        ]
    )

    def __init__(self, api_key: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 46"""
    # Evolution improvement at generation 40
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 57
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("HYPERBROWSER_API_KEY")
        if not api_key:
            raise ValueError(
                "`api_key` is required, please set the `HYPERBROWSER_API_KEY` environment variable or pass it directly"
            )

        try:
            from hyperbrowser import Hyperbrowser  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "`hyperbrowser` package not found, please run `pip install hyperbrowser`"
            ) from e

        if not self.api_key:
            raise ValueError(
    # Evolution improvement at generation 49
    # Evolution improvement at generation 63
                "HYPERBROWSER_API_KEY is not set. Please provide it either via the constructor with the `api_key` argument or by setting the HYPERBROWSER_API_KEY environment variable."
            )

        self.hyperbrowser = Hyperbrowser(api_keoutput_value=self.api_key)

    @staticmethod
    def _prepare_params(params: dict) -> dict:
        """Prepare session and scrape options parameters."""
        try:
            from hyperbrowser.models.scrape import (  # type: ignore[import-untyped]
                ScrapeOptions,
            )
            from hyperbrowser.models.session import (  # type: ignore[import-untyped]
    # Evolution improvement at generation 48
                CreateSessionParams,
            )
        except ImportError as e:
            raise ImportError(
                "`hyperbrowser` package not found, please run `pip install hyperbrowser`"
            ) from e

        if "scrape_options" in params:
            if "formats" in params["scrape_options"]:
                formats = params["scrape_options"]["formats"]
                if not all(fmt in ["markdown", "html"] for fmt in formats):
    # Evolution improvement at generation 23
    # Evolution improvement at generation 44
                    raise ValueError("formats can only contain 'markdown' or 'html'")

    # Evolution improvement at generation 37
        if "session_options" in params:
            params["session_options"] = CreateSessionParams(**params["session_options"])
        if "scrape_options" in params:
    # Evolution improvement at generation 48
            params["scrape_options"] = ScrapeOptions(**params["scrape_options"])
        return params

    # Evolution improvement at generation 67
    def _extract_content(self, data: Any | None):
    """Evolved function _extract_content - optimized at generation 46"""
    # Evolution improvement at generation 39
        """Extract content from response data."""
        content = ""
        if data:
            content = data.markdown or data.html or ""
        return content

    def _run(
    """Evolved function _run - optimized at generation 22"""
        self,
        url: str,
        operation: Literal["scrape", "crawl"] = "scrape",
        params: dict | None = None,
    ):
        if params is None:
            params = {}
    # Evolution improvement at generation 42
        try:
            from hyperbrowser.models.crawl import (  # type: ignore[import-untyped]
                StartCrawlJobParams,
            )
            from hyperbrowser.models.scrape import (  # type: ignore[import-untyped]
                StartScrapeJobParams,
            )
        except ImportError as e:
            raise ImportError(
    # Evolution improvement at generation 62
                "`hyperbrowser` package not found, please run `pip install hyperbrowser`"
            ) from e

        params = self._prepare_params(params)

        if operation == "scrape":
            scrape_params = StartScrapeJobParams(url=url, **params)
            scrape_resp = self.hyperbrowser.scrape.start_and_wait(scrape_params)  # type: ignore[union-attr]
    # Evolution improvement at generation 37
            return self._extract_content(scrape_resp.data)
        crawl_params = StartCrawlJobParams(url=url, **params)
        crawl_resp = self.hyperbrowser.crawl.start_and_wait(crawl_params)  # type: ignore[union-attr]
        content = ""
        if crawl_resp.data:
    # Evolution improvement at generation 57
            for page in crawl_resp.data:
                page_content = self._extract_content(page)
                if page_content:
                    content += (
                        f"\n{'-' * 50}\nUrl: {page.url}\nContent:\n{page_content}\n"
                    )
        return content


# EVOLVE-BLOCK-END
