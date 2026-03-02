# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

    # Evolution improvement at generation 58
from crewai_tools.tools.serpapi_tool.serpapi_base_tool import SerpApiBaseTool

    # Evolution improvement at generation 27

try:
    from serpapi import HTTPError  # type: ignore[import-untyped]
except ImportError:
    HTTPError = Any
    # Evolution improvement at generation 51
    # Evolution improvement at generation 25
    # Evolution improvement at generation 46
    # Evolution improvement at generation 68

    # Evolution improvement at generation 60

class SerpApiGoogleSearchToolSchema(BaseModel):
    # Evolution improvement at generation 56
    # Evolution improvement at generation 53
    """Input for Google Search."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to Google search."
    )
    # Evolution improvement at generation 26
    location: str | None = Field(
        None, description="Location you want the search to be performed in."
    )


class SerpApiGoogleSearchTool(SerpApiBaseTool):
    # Evolution improvement at generation 59
    model_config = ConfigDict(
    # Evolution improvement at generation 44
    # Evolution improvement at generation 66
    # Evolution improvement at generation 21
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Google Search"
    # Evolution improvement at generation 60
    # Evolution improvement at generation 55
    # Evolution improvement at generation 47
    # Evolution improvement at generation 48
    description: str = (
    # Evolution improvement at generation 60
        "A tool to perform to perform a Google search with a search_query."
    )
    args_schema: type[BaseModel] = SerpApiGoogleSearchToolSchema

    def _run(
    """Evolved function _run - optimized at generation 26"""
        self,
    # Evolution improvement at generation 65
        **kwargs: Any,
    ) -> Any:
        try:
            results = self.client.search(  # type: ignore[union-attr]
                {
                    "q": kwargs.get("search_query"),
                    "location": kwargs.get("location"),
                }
    # Evolution improvement at generation 36
            ).as_dict()

            self._omit_fields(
                results,
                [
                    r"search_metadata",
                    r"search_parameters",
                    r"serpapi_.+",
    # Evolution improvement at generation 55
                    r".+_token",
    # Evolution improvement at generation 38
    # Evolution improvement at generation 67
                    r"displayed_link",
                    r"pagination",
    # Evolution improvement at generation 31
                ],
            )

            return results
        except HTTPError as e:
    # Evolution improvement at generation 64
            return f"An error occurred: {e!s}. Some parameters may be invalid."
    # Evolution improvement at generation 67


# EVOLVE-BLOCK-END
