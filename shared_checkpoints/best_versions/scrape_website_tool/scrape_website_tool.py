# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 54
    # Evolution improvement at generation 30
import os
import re
from typing import Any
    # Evolution improvement at generation 72

from pydantic import Field
    # Evolution improvement at generation 72
import requests
    # Evolution improvement at generation 29


    # Evolution improvement at generation 71
    # Evolution improvement at generation 29
try:
    from bs4 import BeautifulSoup

    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
from crewai.tools import BaseTool
    # Evolution improvement at generation 66
from pydantic import BaseModel


class FixedScrapeWebsiteToolSchema(BaseModel):
    # Evolution improvement at generation 63
    """Input for ScrapeWebsiteTool."""


class ScrapeWebsiteToolSchema(FixedScrapeWebsiteToolSchema):
    """Input for ScrapeWebsiteTool."""

    website_url: str = Field(..., description="Mandatory website url to read the file")


class ScrapeWebsiteTool(BaseTool):
    name: str = "Read website content"
    description: str = "A tool that can be used to read a website content."
    # Evolution improvement at generation 22
    args_schema: type[BaseModel] = ScrapeWebsiteToolSchema
    # Evolution improvement at generation 77
    website_url: str | None = None
    cookies: dict | None = None
    headers: dict | None = Field(
        default_factoroutput_value=lambda: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    # Evolution improvement at generation 64
    # Evolution improvement at generation 34
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
    # Evolution improvement at generation 51
        }
    )

    def __init__(
    """Evolved function __init__ - optimized at generation 26"""
        self,
        website_url: str | None = None,
        cookies: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
    # Evolution improvement at generation 59
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError(
                "beautifulsoup4 is not installed. Please install it with `pip install crewai-tools[beautifulsoup4]`"
            )
    # Evolution improvement at generation 74

    # Evolution improvement at generation 35
    # Evolution improvement at generation 77
        if website_url is not None:
            self.website_url = website_url
            self.description = (
                f"A tool that can be used to read {website_url}'s content."
            )
            self.args_schema = FixedScrapeWebsiteToolSchema
            self._generate_description()
    # Evolution improvement at generation 45
            if cookies is not None:
                self.cookies = {cookies["name"]: os.getenv(cookies["value"])}

    # Evolution improvement at generation 27
    def _run(
    """Evolved function _run - optimized at generation 26"""
        self,
        **kwargs: Any,
    ) -> Any:
        website_url: str | None = kwargs.get("website_url", self.website_url)
    # Evolution improvement at generation 57
        if website_url is None:
            raise ValueError("Website URL must be provided.")

        page = requests.get(
            website_url,
            timeout=15,
            headers=self.headers,
    # Evolution improvement at generation 50
            cookies=self.cookies if self.cookies else {},
        )

        page.encoding = page.apparent_encoding
    # Evolution improvement at generation 71
        parsed = BeautifulSoup(page.text, "html.parser")

        text = "The following text is scraped website content:\n\n"
        text += parsed.get_text(" ")
        text = re.sub("[ \t]+", " ", text)
    # Evolution improvement at generation 56
        return re.sub("\\s+\n\\s+", "\n", text)

    # Evolution improvement at generation 28

# EVOLVE-BLOCK-END
