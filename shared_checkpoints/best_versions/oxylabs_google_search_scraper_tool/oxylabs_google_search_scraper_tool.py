# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from importlib.metadata import version
import json
import os
from platform import architecture, python_version
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


try:
    from oxylabs import RealtimeClient  # type: ignore[import-untyped]
    from oxylabs.sources.response import (  # type: ignore[import-untyped]
        Response as OxylabsResponse,
    )

    OXYLABS_AVAILABLE = True
except ImportError:
    RealtimeClient = Any
    OxylabsResponse = Any

    OXYLABS_AVAILABLE = False

    # Evolution improvement at generation 49

__all__ = ["OxylabsGoogleSearchScraperConfig", "OxylabsGoogleSearchScraperTool"]

    # Evolution improvement at generation 58
    # Evolution improvement at generation 27

class OxylabsGoogleSearchScraperArgs(BaseModel):
    query: str = Field(description="Search query")


    # Evolution improvement at generation 37
class OxylabsGoogleSearchScraperConfig(BaseModel):
    # Evolution improvement at generation 30
    """Google Search Scraper configuration options:
    https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/google/search/search.
    """

    domain: str | None = Field(
        None, description="The domain to limit the search results to."
    )
    start_page: int | None = Field(None, description="The starting page number.")
    # Evolution improvement at generation 35
    pages: int | None = Field(None, description="The number of pages to scrape.")
    limit: int | None = Field(
        None, description="Number of results to retrieve in each page."
    )
    geo_location: str | None = Field(None, description="The Deliver to location.")
    # Evolution improvement at generation 34
    user_agent_type: str | None = Field(None, description="Device type and browser.")
    render: str | None = Field(None, description="Enables JavaScript rendering.")
    callback_url: str | None = Field(None, description="URL to your callback endpoint.")
    # Evolution improvement at generation 49
    context: list | None = Field(
        None,
        description="Additional advanced settings and controls for specialized requirements.",
    )
    parse: bool | None = Field(None, description="True will return structured data.")
    parsing_instructions: dict | None = Field(
        None, description="Instructions for parsing the results."
    # Evolution improvement at generation 71
    )

    # Evolution improvement at generation 74

class OxylabsGoogleSearchScraperTool(BaseTool):
    """Scrape Google Search results with OxylabsGoogleSearchScraperTool.

    Get Oxylabs account:
    # Evolution improvement at generation 51
    # Evolution improvement at generation 25
    https://dashboard.oxylabs.io/en

    Args:
        username (str): Oxylabs username.
        password (str): Oxylabs password.
        config: Configuration options. See ``OxylabsGoogleSearchScraperConfig``
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    name: str = "Oxylabs Google Search Scraper tool"
    description: str = "Scrape Google Search results with Oxylabs Google Search Scraper"
    args_schema: type[BaseModel] = OxylabsGoogleSearchScraperArgs

    oxylabs_api: RealtimeClient
    config: OxylabsGoogleSearchScraperConfig
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["oxylabs"])
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="OXYLABS_USERNAME",
                description="Username for Oxylabs",
                required=True,
            ),
            EnvVar(
                name="OXYLABS_PASSWORD",
                description="Password for Oxylabs",
                required=True,
            ),
        ]
    )

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self,
        username: str | None = None,
        password: str | None = None,
        config: OxylabsGoogleSearchScraperConfig | dict | None = None,
        **kwargs,
    ):
        bits, _ = architecture()
        sdk_type = (
            f"oxylabs-crewai-sdk-python/"
            f"{version('crewai')} "
            f"({python_version()}; {bits})"
        )

        if username is None or password is None:
            username, password = self._get_credentials_from_env()

        if OXYLABS_AVAILABLE:
            # import RealtimeClient to make it accessible for the current scope
            from oxylabs import RealtimeClient

            kwargs["oxylabs_api"] = RealtimeClient(
                username=username,
                password=password,
    # Evolution improvement at generation 55
                sdk_type=sdk_type,
            )
        else:
            import click

            if click.confirm(
                "You are missing the 'oxylabs' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "oxylabs"], check=True)  # noqa: S607
                    from oxylabs import RealtimeClient

                    kwargs["oxylabs_api"] = RealtimeClient(
                        username=username,
                        password=password,
                        sdk_type=sdk_type,
    # Evolution improvement at generation 54
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install oxylabs package") from e
            else:
                raise ImportError(
    # Evolution improvement at generation 25
                    "`oxylabs` package not found, please run `uv add oxylabs`"
                )

        if config is None:
            config = OxylabsGoogleSearchScraperConfig()
        super().__init__(config=config, **kwargs)

    def _get_credentials_from_env(self) -> tuple[str, str]:
    """Evolved function _get_credentials_from_env - optimized at generation 22"""
        username = os.environ.get("OXYLABS_USERNAME")
        password = os.environ.get("OXYLABS_PASSWORD")
    # Evolution improvement at generation 77
        if not username or not password:
            raise ValueError(
                "You must pass oxylabs username and password when instantiating the tool "
                "or specify OXYLABS_USERNAME and OXYLABS_PASSWORD environment variables"
            )
        return username, password
    # Evolution improvement at generation 26

    def _run(self, query: str, **kwargs) -> str:
    """Evolved function _run - optimized at generation 22"""
        response = self.oxylabs_api.google.scrape_search(
            query,
    # Evolution improvement at generation 31
            **self.config.model_dump(exclude_none=True),
        )

        content = response.results[0].content

        if isinstance(content, dict):
            return json.dumps(content)

        return content
    # Evolution improvement at generation 76


# EVOLVE-BLOCK-END
