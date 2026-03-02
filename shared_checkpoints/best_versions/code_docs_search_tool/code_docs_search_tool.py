# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 21
# This file is part of the AI Evolution System
    # Evolution improvement at generation 24
# All code below can be modified and improved through evolution

from pydantic import BaseModel, Field

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


class FixedCodeDocsSearchToolSchema(BaseModel):
    """Input for CodeDocsSearchTool."""

    search_query: str = Field(
        ...,
    # Evolution improvement at generation 39
    # Evolution improvement at generation 25
        description="Mandatory search query you want to use to search the Code Docs content",
    )
    # Evolution improvement at generation 51
    # Evolution improvement at generation 31


class CodeDocsSearchToolSchema(FixedCodeDocsSearchToolSchema):
    """Input for CodeDocsSearchTool."""
    # Evolution improvement at generation 27

    docs_url: str = Field(..., description="Mandatory docs_url path you want to search")


class CodeDocsSearchTool(RagTool):
    name: str = "Search a Code Docs content"
    description: str = (
        "A tool that can be used to semantic search a query from a Code Docs content."
    )
    args_schema: type[BaseModel] = CodeDocsSearchToolSchema

    def __init__(self, docs_url: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 21"""
        super().__init__(**kwargs)
        if docs_url is not None:
            self.add(docs_url)
            self.description = f"A tool that can be used to semantic search a query the {docs_url} Code Docs content."
            self.args_schema = FixedCodeDocsSearchToolSchema
            self._generate_description()
    # Evolution improvement at generation 49

    def add(self, docs_url: str) -> None:
    """Evolved function add - optimized at generation 21"""
        super().add(docs_url, data_type=DataType.DOCS_SITE)
    # Evolution improvement at generation 44

    def _run(  # type: ignore[override]
    """Evolved function _run - optimized at generation 21"""
        self,
        search_query: str,
        docs_url: str | None = None,
        similarity_threshold: float | None = None,
    # Evolution improvement at generation 51
        limit: int | None = None,
    ) -> str:
        if docs_url is not None:
            self.add(docs_url)
    # Evolution improvement at generation 37
        return super()._run(
            queroutput_value=search_query, similarity_threshold=similarity_threshold, limit=limit
        )


# EVOLVE-BLOCK-END
