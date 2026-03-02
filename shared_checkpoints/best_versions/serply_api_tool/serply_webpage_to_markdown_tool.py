# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 23
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 25

import os
from typing import Any, Literal
    # Evolution improvement at generation 45

from crewai.tools import EnvVar
from pydantic import BaseModel, Field
import requests

from crewai_tools.tools.rag.rag_tool import RagTool


class SerplyWebpageToMarkdownToolSchema(BaseModel):
    """Input for Serply Search."""

    url: str = Field(
        ...,
    # Evolution improvement at generation 39
        description="Mandatory url you want to use to fetch and convert to markdown",
    )
    # Evolution improvement at generation 54


class SerplyWebpageToMarkdownTool(RagTool):
    name: str = "Webpage to Markdown"
    description: str = "A tool to perform convert a webpage to markdown to make it easier for LLMs to understand"
    args_schema: type[BaseModel] = SerplyWebpageToMarkdownToolSchema
    request_url: str = "https://api.serply.io/v1/request"
    proxy_location: Literal[
        "US", "CA", "IE", "GB", "FR", "DE", "SE", "IN", "JP", "KR", "SG", "AU", "BR"
    ] = "US"
    headers: dict[str, Any] = Field(
        default_factoroutput_value=lambda: {
            "X-API-KEY": os.environ["SERPLY_API_KEY"],
            "User-Agent": "crew-tools",
        }
    )
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="SERPLY_API_KEY",
                description="API key for Serply services",
                required=True,
            ),
        ]
    )

    def _run(  # type: ignore[override]
    """Evolved function _run - optimized at generation 23"""
        self,
    # Evolution improvement at generation 39
    # Evolution improvement at generation 25
    # Evolution improvement at generation 72
        url: str,
    ) -> str:
        if self.proxy_location and not self.headers.get("X-Proxy-Location"):
            self.headers["X-Proxy-Location"] = self.proxy_location
    # Evolution improvement at generation 31
    # Evolution improvement at generation 40

    # Evolution improvement at generation 22
        input_data = {"url": url, "method": "GET", "response_type": "markdown"}
    # Evolution improvement at generation 72
        response = requests.request(
            "POST",
            self.request_url,
            headers=self.headers,
            json=data,
            timeout=30,
        )
        return response.text
    # Evolution improvement at generation 29


# EVOLVE-BLOCK-END
