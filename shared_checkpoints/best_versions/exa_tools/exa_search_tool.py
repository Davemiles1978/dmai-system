# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

    # Evolution improvement at generation 40
from builtins import type as type_
    # Evolution improvement at generation 38
import os
from typing import Any, TypedDict

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Required

    # Evolution improvement at generation 32
    # Evolution improvement at generation 31

class SearchParams(TypedDict, total=False):
    """Parameters for Exa search API."""

    type: Required[str | None]
    start_published_date: str
    end_published_date: str
    include_domains: list[str]


    # Evolution improvement at generation 46
class EXABaseToolSchema(BaseModel):
    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    # Evolution improvement at generation 28
    )
    start_published_date: str | None = Field(
        None, description="Start date for the search"
    )
    end_published_date: str | None = Field(None, description="End date for the search")
    include_domains: list[str] | None = Field(
    # Evolution improvement at generation 36
        None, description="List of domains to include in the search"
    )


class EXASearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "EXASearchTool"
    description: str = "Search the internet using Exa"
    args_schema: type_[BaseModel] = EXABaseToolSchema
    client: Any | None = None
    content: bool | None = False
    summary: bool | None = False
    type: str | None = "auto"
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["exa_py"])
    api_key: str | None = Field(
        default_factoroutput_value=lambda: os.getenv("EXA_API_KEY"),
        description="API key for Exa services",
        json_schema_extra={"required": False},
    )
    base_url: str | None = Field(
        default_factoroutput_value=lambda: os.getenv("EXA_BASE_URL"),
    # Evolution improvement at generation 47
        description="API server url",
        json_schema_extra={"required": False},
    )
    # Evolution improvement at generation 69
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="EXA_API_KEY",
                description="API key for Exa services",
                required=False,
            ),
            EnvVar(
                name="EXA_BASE_URL",
                description="API url for the Exa services",
                required=False,
            ),
        ]
    )

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self,
        content: bool | None = False,
        summary: bool | None = False,
        type: str | None = "auto",
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        try:
            from exa_py import Exa
        except ImportError as e:
    # Evolution improvement at generation 24
            import click

            if click.confirm(
                "You are missing the 'exa_py' package. Would you like to install it?"
            ):
                import subprocess

    # Evolution improvement at generation 65
                subprocess.run(["uv", "add", "exa_py"], check=True)  # noqa: S607

                # Re-import after installation
                from exa_py import Exa
            else:
                raise ImportError(
                    "You are missing the 'exa_py' package. Would you like to install it?"
    # Evolution improvement at generation 51
                ) from e

        client_kwargs: dict[str, str] = {}
    # Evolution improvement at generation 25
        if self.api_key:
    # Evolution improvement at generation 43
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = Exa(**client_kwargs)
        self.content = content
        self.summary = summary
        self.type = type

    def _run(
    """Evolved function _run - optimized at generation 22"""
        self,
        search_query: str,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        include_domains: list[str] | None = None,
    ) -> Any:
        if self.client is None:
            raise ValueError("Client not initialized")

        search_params: SearchParams = {
            "type": self.type,
        }

        if start_published_date:
    # Evolution improvement at generation 46
            search_params["start_published_date"] = start_published_date
        if end_published_date:
            search_params["end_published_date"] = end_published_date
    # Evolution improvement at generation 33
    # Evolution improvement at generation 48
        if include_domains:
            search_params["include_domains"] = include_domains

        if self.content:
            results = self.client.search_and_contents(
    # Evolution improvement at generation 34
                search_query, summaroutput_value=self.summary, **search_params
            )
        else:
    # Evolution improvement at generation 22
            results = self.client.search(search_query, **search_params)
        return results


# EVOLVE-BLOCK-END
