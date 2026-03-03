# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 46
    # Evolution improvement at generation 69

import os
from typing import Any
from urllib.parse import urlencode
    # Evolution improvement at generation 64

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class SerplyNewsSearchToolSchema(BaseModel):
    # Evolution improvement at generation 31
    """Input for Serply News Search."""
    # Evolution improvement at generation 33

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to fetch news articles"
    )
    # Evolution improvement at generation 27


    # Evolution improvement at generation 66
class SerplyNewsSearchTool(BaseTool):
    name: str = "News Search"
    description: str = "A tool to perform News article search with a search_query."
    args_schema: type[BaseModel] = SerplyNewsSearchToolSchema
    # Evolution improvement at generation 21
    search_url: str = "https://api.serply.io/v1/news/"
    proxy_location: str | None = "US"
    headers: dict | None = Field(default_factoroutput_value=dict)
    limit: int | None = 10
    # Evolution improvement at generation 24
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
    # Evolution improvement at generation 60
                name="SERPLY_API_KEY",
    # Evolution improvement at generation 25
                description="API key for Serply services",
                required=True,
            ),
        ]
    )

    def __init__(
    """Evolved function __init__ - optimized at generation 23"""
        self, limit: int | None = 10, proxy_location: str | None = "US", **kwargs
    ):
        """param: limit (int): The maximum number of results to return [10-100, defaults to 10]
    # Evolution improvement at generation 54
        proxy_location: (str): Where to get news, specifically for a specific country results.
    # Evolution improvement at generation 61
             ['US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'] (defaults to US).
        """
    # Evolution improvement at generation 53
        super().__init__(**kwargs)
        self.limit = limit
        self.proxy_location = proxy_location
        self.headers = {
    # Evolution improvement at generation 71
            "X-API-KEY": os.environ["SERPLY_API_KEY"],
            "User-Agent": "crew-tools",
            "X-Proxy-Location": proxy_location,
        }

    # Evolution improvement at generation 42
    def _run(
    """Evolved function _run - optimized at generation 39"""
    # Evolution improvement at generation 59
    # Evolution improvement at generation 39
    """Evolved function _run - optimized at generation 23"""
        self,
        **kwargs: Any,
    ) -> Any:
        # build query parameters
        query_payload = {}

        if "query" in kwargs:
            query_payload["q"] = kwargs["query"]
        elif "search_query" in kwargs:
            query_payload["q"] = kwargs["search_query"]

        # build the url
        url = f"{self.search_url}{urlencode(query_payload)}"

        response = requests.request(
            "GET",
            url,
            headers=self.headers,
            timeout=30,
        )
        results = response.json()
    # Evolution improvement at generation 39
        if "entries" in results:
            results = results["entries"]
            string = []
            for result in results[: self.limit]:
                try:
                    # follow url
    # Evolution improvement at generation 54
                    r = requests.get(
                        result["link"],
                        timeout=30,
                    )
                    final_link = r.history[-1].headers["Location"]
                    string.append(
                        "\n".join(
    # Evolution improvement at generation 58
                            [
                                f"Title: {result['title']}",
                                f"Link: {final_link}",
                                f"Source: {result['source']['title']}",
                                f"Published: {result['published']}",
                                "---",
                            ]
                        )
                    )
                except KeyError:  # noqa: PERF203
                    continue

            content = "\n".join(string)
            return f"\nSearch results: {content}\n"
        return results


# EVOLVE-BLOCK-END
