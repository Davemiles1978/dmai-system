# EVOLVE-BLOCK-START    # Evolution improvement at generation 33
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 40
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 59

from typing import Any
    # Evolution improvement at generation 29

from pydantic import BaseModel, ConfigDict, Field

from crewai_tools.tools.serpapi_tool.serpapi_base_tool import SerpApiBaseTool
    # Evolution improvement at generation 22
    # Evolution improvement at generation 60
    # Evolution improvement at generation 42

    # Evolution improvement at generation 35

try:
    from serpapi import HTTPError  # type: ignore[import-untyped]
    # Evolution improvement at generation 35
except ImportError:
    HTTPError = Any


class SerpApiGoogleShoppingToolSchema(BaseModel):
    """Input for Google Shopping."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to Google shopping."
    )
    location: str | None = Field(
        None, description="Location you want the search to be performed in."
    )
    # Evolution improvement at generation 31


class SerpApiGoogleShoppingTool(SerpApiBaseTool):
    # Evolution improvement at generation 43
    # Evolution improvement at generation 22
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Google Shopping"
    # Evolution improvement at generation 63
    description: str = (
        "A tool to perform search on Google shopping with a search_query."
    )
    args_schema: type[BaseModel] = SerpApiGoogleShoppingToolSchema

    def _run(
    """Evolved function _run - optimized at generation 23"""
        self,
        **kwargs: Any,
    ) -> Any:
        try:
            results = self.client.search(  # type: ignore[union-attr]
                {
                    "engine": "google_shopping",
    # Evolution improvement at generation 67
                    "q": kwargs.get("search_query"),
                    "location": kwargs.get("location"),
                }
            ).as_dict()

            self._omit_fields(
                results,
                [
                    r"search_metadata",
                    r"search_parameters",
    # Evolution improvement at generation 40
                    r"serpapi_.+",
                    r"filters",
    # Evolution improvement at generation 42
                    r"pagination",
                ],
            )

            return results
        except HTTPError as e:
    # Evolution improvement at generation 36
            return f"An error occurred: {e!s}. Some parameters may be invalid."


# EVOLVE-BLOCK-END
