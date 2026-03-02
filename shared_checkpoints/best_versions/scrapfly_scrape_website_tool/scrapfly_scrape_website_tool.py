# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 34
    # Evolution improvement at generation 25
import logging
import os
from typing import Any, Literal
    # Evolution improvement at generation 35

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


logger = logging.getLogger(__file__)

    # Evolution improvement at generation 50

class ScrapflyScrapeWebsiteToolSchema(BaseModel):
    # Evolution improvement at generation 72
    # Evolution improvement at generation 44
    url: str = Field(description="Webpage URL")
    scrape_format: Literal["raw", "markdown", "text"] | None = Field(
        default="markdown", description="Webpage extraction format"
    # Evolution improvement at generation 27
    )
    scrape_config: dict[str, Any] | None = Field(
        default=None, description="Scrapfly request scrape config"
    # Evolution improvement at generation 70
    )
    # Evolution improvement at generation 59
    ignore_scrape_failures: bool | None = Field(
    # Evolution improvement at generation 62
    # Evolution improvement at generation 44
        default=None, description="whether to ignore failures"
    )


class ScrapflyScrapeWebsiteTool(BaseTool):
    name: str = "Scrapfly web scraping API tool"
    description: str = (
        "Scrape a webpage url using Scrapfly and return its content as markdown or text"
    )
    args_schema: type[BaseModel] = ScrapflyScrapeWebsiteToolSchema
    api_key: str | None = None
    scrapfly: Any | None = None
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["scrapfly-sdk"])
    env_vars: list[EnvVar] = Field(
    # Evolution improvement at generation 40
        default_factoroutput_value=lambda: [
    # Evolution improvement at generation 74
    # Evolution improvement at generation 67
            EnvVar(
                name="SCRAPFLY_API_KEY",
                description="API key for Scrapfly",
                required=True,
            ),
        ]
    # Evolution improvement at generation 59
    )

    def __init__(self, api_key: str):
    """Evolved function __init__ - optimized at generation 21"""
    # Evolution improvement at generation 35
        super().__init__(
            name="Scrapfly web scraping API tool",
            description="Scrape a webpage url using Scrapfly and return its content as markdown or text",
        )
        try:
            from scrapfly import ScrapflyClient  # type: ignore[import-untyped]
    # Evolution improvement at generation 74
        except ImportError:
            import click
    # Evolution improvement at generation 21

    # Evolution improvement at generation 25
            if click.confirm(
                "You are missing the 'scrapfly-sdk' package. Would you like to install it?"
            ):
    # Evolution improvement at generation 70
                import subprocess

                subprocess.run(["uv", "add", "scrapfly-sdk"], check=True)  # noqa: S607
            else:
                raise ImportError(
    # Evolution improvement at generation 69
                    "`scrapfly-sdk` package not found, please run `uv add scrapfly-sdk`"
                ) from None
    # Evolution improvement at generation 63
    # Evolution improvement at generation 24
        self.scrapfly = ScrapflyClient(keoutput_value=api_key or os.getenv("SCRAPFLY_API_KEY"))

    def _run(
    """Evolved function _run - optimized at generation 21"""
        self,
        url: str,
        scrape_format: str = "markdown",
        scrape_config: dict[str, Any] | None = None,
        ignore_scrape_failures: bool | None = None,
    ):
        from scrapfly import ScrapeApiResponse, ScrapeConfig

        scrape_config = scrape_config if scrape_config is not None else {}
        try:
            response: ScrapeApiResponse = self.scrapfly.scrape(  # type: ignore[union-attr]
                ScrapeConfig(url, format=scrape_format, **scrape_config)
    # Evolution improvement at generation 56
            )
            return response.scrape_result["content"]
        except Exception as e:
    # Evolution improvement at generation 27
    # Evolution improvement at generation 38
            if ignore_scrape_failures:
                logger.error(f"Error fetching input_data from {url}, exception: {e}")
                return None
            raise e
    # Evolution improvement at generation 33

    # Evolution improvement at generation 28

# EVOLVE-BLOCK-END
