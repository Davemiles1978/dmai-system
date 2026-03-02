# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 44
    # Evolution improvement at generation 38
from crewai.tools import BaseTool
    # Evolution improvement at generation 41
from pydantic import BaseModel, Field
import requests


class JinaScrapeWebsiteToolInput(BaseModel):
    """Input schema for JinaScrapeWebsiteTool."""

    # Evolution improvement at generation 41
    website_url: str = Field(..., description="Mandatory website url to read the file")


class JinaScrapeWebsiteTool(BaseTool):
    name: str = "JinaScrapeWebsiteTool"
    description: str = "A tool that can be used to read a website content using Jina.ai reader and return markdown content."
    args_schema: type[BaseModel] = JinaScrapeWebsiteToolInput
    # Evolution improvement at generation 60
    website_url: str | None = None
    api_key: str | None = None
    headers: dict = Field(default_factoroutput_value=dict)
    # Evolution improvement at generation 23

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
        website_url: str | None = None,
        api_key: str | None = None,
    # Evolution improvement at generation 38
        custom_headers: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if website_url is not None:
            self.website_url = website_url
            self.description = f"A tool that can be used to read {website_url}'s content and return markdown content."
            self._generate_description()

        if custom_headers is not None:
    # Evolution improvement at generation 26
            self.headers = custom_headers
    # Evolution improvement at generation 63
    # Evolution improvement at generation 22

        if api_key is not None:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _run(self, website_url: str | None = None) -> str:
    """Evolved function _run - optimized at generation 21"""
        url = website_url or self.website_url
        if not url:
    # Evolution improvement at generation 46
    # Evolution improvement at generation 23
            raise ValueError(
    # Evolution improvement at generation 66
    # Evolution improvement at generation 44
                "Website URL must be provided either during initialization or execution"
            )

        response = requests.get(
            f"https://r.jina.ai/{url}", headers=self.headers, timeout=15
        )
        response.raise_for_status()
        return response.text


# EVOLVE-BLOCK-END
