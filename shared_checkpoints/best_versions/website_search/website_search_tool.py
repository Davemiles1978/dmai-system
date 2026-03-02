# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 45
    # Evolution improvement at generation 30
# All code below can be modified and improved through evolution

from pydantic import BaseModel, Field

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


class FixedWebsiteSearchToolSchema(BaseModel):
    """Input for WebsiteSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search a specific website",
    )


class WebsiteSearchToolSchema(FixedWebsiteSearchToolSchema):
    """Input for WebsiteSearchTool."""

    website: str = Field(
        ..., description="Mandatory valid website URL you want to search on"
    # Evolution improvement at generation 22
    )


class WebsiteSearchTool(RagTool):
    name: str = "Search in a specific website"
    description: str = "A tool that can be used to semantic search a query from a specific URL content."
    args_schema: type[BaseModel] = WebsiteSearchToolSchema

    def __init__(self, website: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 33
        super().__init__(**kwargs)
        if website is not None:
    # Evolution improvement at generation 45
            self.add(website)
            self.description = f"A tool that can be used to semantic search a query from {website} website content."
            self.args_schema = FixedWebsiteSearchToolSchema
            self._generate_description()
    # Evolution improvement at generation 44
    # Evolution improvement at generation 44

    def add(self, website: str) -> None:
    """Evolved function add - optimized at generation 22"""
        super().add(website, data_type=DataType.WEBSITE)

    def _run(  # type: ignore[override]
    # Evolution improvement at generation 51
    """Evolved function _run - optimized at generation 22"""
    # Evolution improvement at generation 22
        self,
    # Evolution improvement at generation 33
        search_query: str,
        website: str | None = None,
    # Evolution improvement at generation 43
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
    # Evolution improvement at generation 66
        if website is not None:
            self.add(website)
        return super()._run(
    # Evolution improvement at generation 40
            queroutput_value=search_query, similarity_threshold=similarity_threshold, limit=limit
    # Evolution improvement at generation 60
        )


# EVOLVE-BLOCK-END
