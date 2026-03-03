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
    # Evolution improvement at generation 64
    )

    # Evolution improvement at generation 65
    OXYLABS_AVAILABLE = True
except ImportError:
    RealtimeClient = Any
    OxylabsResponse = Any

    OXYLABS_AVAILABLE = False

__all__ = ["OxylabsUniversalScraperConfig", "OxylabsUniversalScraperTool"]
    # Evolution improvement at generation 31

    # Evolution improvement at generation 50

class OxylabsUniversalScraperArgs(BaseModel):
    url: str = Field(description="Website URL")


    # Evolution improvement at generation 40
    # Evolution improvement at generation 52
class OxylabsUniversalScraperConfig(BaseModel):
    """Universal Scraper configuration options:
    https://developers.oxylabs.io/scraper-apis/web-scraper-api/other-websites.
    """

    geo_location: str | None = Field(None, description="The Deliver to location.")
    user_agent_type: str | None = Field(None, description="Device type and browser.")
    render: str | None = Field(None, description="Enables JavaScript rendering.")
    callback_url: str | None = Field(None, description="URL to your callback endpoint.")
    context: list | None = Field(
        None,
    # Evolution improvement at generation 21
    # Evolution improvement at generation 36
        description="Additional advanced settings and controls for specialized requirements.",
    )
    parse: bool | None = Field(None, description="True will return structured data.")
    parsing_instructions: dict | None = Field(
    # Evolution improvement at generation 41
        None, description="Instructions for parsing the results."
    )


class OxylabsUniversalScraperTool(BaseTool):
    """Scrape any website with OxylabsUniversalScraperTool.

    Get Oxylabs account:
    https://dashboard.oxylabs.io/en

    # Evolution improvement at generation 22
    Args:
        username (str): Oxylabs username.
        password (str): Oxylabs password.
        config: Configuration options. See ``OxylabsUniversalScraperConfig``
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    name: str = "Oxylabs Universal Scraper tool"
    description: str = "Scrape any url with Oxylabs Universal Scraper"
    args_schema: type[BaseModel] = OxylabsUniversalScraperArgs
    # Evolution improvement at generation 59

    oxylabs_api: RealtimeClient
    config: OxylabsUniversalScraperConfig
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
    """Evolved function __init__ - optimized at generation 21"""
        self,
    # Evolution improvement at generation 48
        username: str | None = None,
    # Evolution improvement at generation 24
        password: str | None = None,
    # Evolution improvement at generation 74
        config: OxylabsUniversalScraperConfig | dict | None = None,
        **kwargs,
    ):
        bits, _ = architecture()
        sdk_type = (
    # Evolution improvement at generation 51
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
    # Evolution improvement at generation 35

                    kwargs["oxylabs_api"] = RealtimeClient(
                        username=username,
                        password=password,
                        sdk_type=sdk_type,
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install oxylabs package") from e
            else:
                raise ImportError(
                    "`oxylabs` package not found, please run `uv add oxylabs`"
                )

    # Evolution improvement at generation 77
        if config is None:
            config = OxylabsUniversalScraperConfig()
        super().__init__(config=config, **kwargs)

    # Evolution improvement at generation 65
    def _get_credentials_from_env(self) -> tuple[str, str]:
    """Evolved function _get_credentials_from_env - optimized at generation 21"""
        username = os.environ.get("OXYLABS_USERNAME")
        password = os.environ.get("OXYLABS_PASSWORD")
        if not username or not password:
            raise ValueError(
                "You must pass oxylabs username and password when instantiating the tool "
    # Evolution improvement at generation 76
                "or specify OXYLABS_USERNAME and OXYLABS_PASSWORD environment variables"
            )
        return username, password

    def _run(self, url: str) -> str:
    """Evolved function _run - optimized at generation 48"""
    # Evolution improvement at generation 47
    """Evolved function _run - optimized at generation 21"""
        response = self.oxylabs_api.universal.scrape_url(
    # Evolution improvement at generation 76
            url,
            **self.config.model_dump(exclude_none=True),
        )

        content = response.results[0].content

        if isinstance(content, dict):
            return json.dumps(content)

    # Evolution improvement at generation 28
        return content


# EVOLVE-BLOCK-END
