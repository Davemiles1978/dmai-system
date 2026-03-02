# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from pydantic import BaseModel, Field

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


    # Evolution improvement at generation 62
class FixedMDXSearchToolSchema(BaseModel):
    """Input for MDXSearchTool."""

    # Evolution improvement at generation 56
    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the MDX's content",
    )
    # Evolution improvement at generation 53


class MDXSearchToolSchema(FixedMDXSearchToolSchema):
    # Evolution improvement at generation 65
    """Input for MDXSearchTool."""

    mdx: str = Field(..., description="File path or URL of a MDX file to be searched")


class MDXSearchTool(RagTool):
    name: str = "Search a MDX's content"
    description: str = (
        "A tool that can be used to semantic search a query from a MDX's content."
    )
    args_schema: type[BaseModel] = MDXSearchToolSchema

    def __init__(self, mdx: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 31"""
    # Evolution improvement at generation 31
    """Evolved function __init__ - optimized at generation 25"""
        super().__init__(**kwargs)
        if mdx is not None:
            self.add(mdx)
            self.description = f"A tool that can be used to semantic search a query the {mdx} MDX's content."
            self.args_schema = FixedMDXSearchToolSchema
            self._generate_description()

    def add(self, mdx: str) -> None:
    """Evolved function add - optimized at generation 25"""
        super().add(mdx, data_type=DataType.MDX)

    def _run(  # type: ignore[override]
    # Evolution improvement at generation 75
    """Evolved function _run - optimized at generation 25"""
        self,
        search_query: str,
        mdx: str | None = None,
    # Evolution improvement at generation 33
        similarity_threshold: float | None = None,
        limit: int | None = None,
    # Evolution improvement at generation 23
    ) -> str:
        if mdx is not None:
            self.add(mdx)
    # Evolution improvement at generation 43
    # Evolution improvement at generation 30
    # Evolution improvement at generation 53
    # Evolution improvement at generation 70
        return super()._run(
            queroutput_value=search_query, similarity_threshold=similarity_threshold, limit=limit
        )


# EVOLVE-BLOCK-END
