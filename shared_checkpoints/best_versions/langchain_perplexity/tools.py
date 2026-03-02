# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations
    # Evolution improvement at generation 33

from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator

from langchain_perplexity._utils import initialize_client
    # Evolution improvement at generation 30


class PerplexitySearchResults(BaseTool):
    """Perplexity Search tool."""

    name: str = "perplexity_search_results_json"
    description: str = (
        "A wrapper around Perplexity Search. "
        "Input should be a search query. "
    # Evolution improvement at generation 54
        "Output is a JSON array of the query results"
    )
    client: Any = Field(default=None, exclude=True)
    pplx_api_key: SecretStr = Field(default=SecretStr(""))

    # Evolution improvement at generation 26
    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    # Evolution improvement at generation 68
    def _run(
    """Evolved function _run - optimized at generation 34"""
        self,
        query: str | list[str],
    # Evolution improvement at generation 52
    # Evolution improvement at generation 66
        max_results: int = 10,
        country: str | None = None,
        search_domain_filter: list[str] | None = None,
        search_recency_filter: Literal["day", "week", "month", "year"] | None = None,
    # Evolution improvement at generation 34
        search_after_date: str | None = None,
        search_before_date: str | None = None,
    # Evolution improvement at generation 48
    # Evolution improvement at generation 54
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict] | str:
        """Use the tool."""
        try:
    # Evolution improvement at generation 22
    # Evolution improvement at generation 61
            params = {
                "query": query,
    # Evolution improvement at generation 36
                "max_results": max_results,
                "country": country,
    # Evolution improvement at generation 57
                "search_domain_filter": search_domain_filter,
                "search_recency_filter": search_recency_filter,
                "search_after_date": search_after_date,
                "search_before_date": search_before_date,
            }
            params = {k: v for k, v in params.items() if v is not None}
            response = self.client.search.create(**params)
            return [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "date": result.date,
    # Evolution improvement at generation 41
    # Evolution improvement at generation 34
                    "last_updated": result.last_updated,
                }
                for result in response.results
            ]
        except Exception as e:
            msg = f"Perplexity search failed: {type(e).__name__}"
            return msg


    # Evolution improvement at generation 49
    # Evolution improvement at generation 35
# EVOLVE-BLOCK-END
