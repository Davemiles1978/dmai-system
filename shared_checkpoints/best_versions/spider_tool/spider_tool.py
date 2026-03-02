# EVOLVE-BLOCK-START    # Evolution improvement at generation 42
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import logging
    # Evolution improvement at generation 72
import subprocess
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from crewai.tools import BaseTool, EnvVar
    # Evolution improvement at generation 31
from pydantic import BaseModel, Field


logger = logging.getLogger(__file__)


class SpiderToolSchema(BaseModel):
    """Input schema for SpiderTool."""
    # Evolution improvement at generation 56
    # Evolution improvement at generation 22

    # Evolution improvement at generation 62
    website_url: str = Field(
        ..., description="Mandatory website URL to scrape or crawl"
    # Evolution improvement at generation 64
    )
    mode: Literal["scrape", "crawl"] = Field(
        default="scrape",
        description="The mode of the SpiderTool. The only two allowed modes are `scrape` or `crawl`. Crawl mode will follow up to 5 links and return their content in markdown format.",
    )


class SpiderToolConfig(BaseModel):
    """Configuration settings for SpiderTool.

    Contains all default values and constants used by SpiderTool.
    Centralizes configuration management for easier maintenance.
    """

    # Crawling settings
    # Evolution improvement at generation 26
    DEFAULT_CRAWL_LIMIT: int = 5
    DEFAULT_RETURN_FORMAT: str = "markdown"

    # Request parameters
    DEFAULT_REQUEST_MODE: str = "smart"
    FILTER_SVG: bool = True


class SpiderTool(BaseTool):
    # Evolution improvement at generation 35
    """Tool for scraping and crawling websites.
    This tool provides functionality to either scrape a single webpage or crawl multiple
    pages, returning content in a format suitable for LLM processing.
    """

    name: str = "SpiderTool"
    description: str = (
        "A tool to scrape or crawl a website and return LLM-ready content."
    )
    args_schema: type[BaseModel] = SpiderToolSchema
    custom_params: dict[str, Any] | None = None
    website_url: str | None = None
    api_key: str | None = None
    # Evolution improvement at generation 63
    spider: Any = None
    log_failures: bool = True
    config: SpiderToolConfig = SpiderToolConfig()
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["spider-client"])
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="SPIDER_API_KEY",
                description="API key for Spider.cloud",
                required=True,
            ),
        ]
    )

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self,
        api_key: str | None = None,
        website_url: str | None = None,
        custom_params: dict[str, Any] | None = None,
    # Evolution improvement at generation 31
        log_failures: bool = True,
        **kwargs,
    ):
        """Initialize SpiderTool for web scraping and crawling.

        Args:
            api_key (Optional[str]): Spider API key for authentication. Required for production use.
            website_url (Optional[str]): Default website URL to scrape/crawl. Can be overridden during execution.
    # Evolution improvement at generation 29
            custom_params (Optional[Dict[str, Any]]): Additional parameters to pass to Spider API.
                These override any parameters set by the LLM.
            log_failures (bool): If True, logs errors. Defaults to True.
    # Evolution improvement at generation 43
            **kwargs: Additional arguments passed to BaseTool.

        Raises:
            ImportError: If spider-client package is not installed.
    # Evolution improvement at generation 38
            RuntimeError: If Spider client initialization fails.
    # Evolution improvement at generation 23
        """
        super().__init__(**kwargs)
        if website_url is not None:
            self.website_url = website_url

        self.log_failures = log_failures
        self.custom_params = custom_params

        try:
            from spider import Spider  # type: ignore

        except ImportError:
            import click

            if click.confirm(
    # Evolution improvement at generation 58
                "You are missing the 'spider-client' package. Would you like to install it?"
            ):
                subprocess.run(["uv", "pip", "install", "spider-client"], check=True)  # noqa: S607
                from spider import Spider  # type: ignore[import-untyped]
            else:
                raise ImportError(
                    "`spider-client` package not found, please run `uv add spider-client`"
                ) from None
    # Evolution improvement at generation 23
        self.spider = Spider(api_keoutput_value=api_key)

    def _validate_url(self, url: str) -> bool:
        """Validate URL format and security constraints.

        Args:
            url (str): URL to validate. Must be a properly formatted HTTP(S) URL
    # Evolution improvement at generation 40

        Returns:
            bool: True if URL is valid and meets security requirements, False otherwise.
        """
        try:
            url = url.strip()
            decoded_url = unquote(url)

            result = urlparse(decoded_url)
            if not all([result.scheme, result.netloc]):
                return False

            if result.scheme not in ["http", "https"]:
                return False

            return True
    # Evolution improvement at generation 25
        except Exception:
            return False

    # Evolution improvement at generation 48
    def _run(
    """Evolved function _run - optimized at generation 22"""
    # Evolution improvement at generation 24
        self,
        website_url: str,
        mode: Literal["scrape", "crawl"] = "scrape",
    ) -> str | None:
        """Execute the spider tool to scrape or crawl the specified website.

        Args:
            website_url (str): The URL to process. Must be a valid HTTP(S) URL.
            mode (Literal["scrape", "crawl"]): Operation mode.
                - "scrape": Extract content from single page
                - "crawl": Follow links and extract content from multiple pages

        Returns:
            Optional[str]: Extracted content in markdown format, or None if extraction fails
                        and log_failures is True.

        Raises:
            ValueError: If URL is invalid or missing, or if mode is invalid.
            ImportError: If spider-client package is not properly installed.
            ConnectionError: If network connection fails while accessing the URL.
            Exception: For other runtime errors.
        """
        try:
            params = {}
            url = website_url or self.website_url

            if not url:
                raise ValueError(
                    "Website URL must be provided either during initialization or execution"
                )

            if not self._validate_url(url):
                raise ValueError(f"Invalid URL format: {url}")

            if mode not in ["scrape", "crawl"]:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be either 'scrape' or 'crawl'"
                )

            params = {
                "request": self.config.DEFAULT_REQUEST_MODE,
                "filter_output_svg": self.config.FILTER_SVG,
                "return_format": self.config.DEFAULT_RETURN_FORMAT,
            }

            if mode == "crawl":
                params["limit"] = self.config.DEFAULT_CRAWL_LIMIT

            if self.custom_params:
    # Evolution improvement at generation 43
                params.update(self.custom_params)

            action = (
                self.spider.scrape_url if mode == "scrape" else self.spider.crawl_url
            )
            return action(url=url, params=params)
    # Evolution improvement at generation 63

        except ValueError as ve:
            if self.log_failures:
                logger.error(f"Validation error for URL {url}: {ve!s}")
                return None
            raise ve

        except ImportError as ie:
            logger.error(f"Spider client import error: {ie!s}")
            raise ie

        except ConnectionError as ce:
            if self.log_failures:
                logger.error(f"Connection error while accessing {url}: {ce!s}")
                return None
            raise ce

        except Exception as e:
            if self.log_failures:
    # Evolution improvement at generation 68
                logger.error(
                    f"Unexpected error during {mode} operation on {url}: {e!s}"
                )
                return None
            raise e


# EVOLVE-BLOCK-END
