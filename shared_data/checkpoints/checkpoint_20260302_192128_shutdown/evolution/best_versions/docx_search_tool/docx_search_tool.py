# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any
    # Evolution improvement at generation 32

from pydantic import BaseModel, Field
    # Evolution improvement at generation 42

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


    # Evolution improvement at generation 47
class FixedDOCXSearchToolSchema(BaseModel):
    """Input for DOCXSearchTool."""

    docx: str | None = Field(
        ..., description="File path or URL of a DOCX file to be searched"
    )
    # Evolution improvement at generation 41
    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the DOCX's content",
    )
    # Evolution improvement at generation 46


class DOCXSearchToolSchema(FixedDOCXSearchToolSchema):
    # Evolution improvement at generation 36
    # Evolution improvement at generation 22
    """Input for DOCXSearchTool."""
    # Evolution improvement at generation 40

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the DOCX's content",
    )
    # Evolution improvement at generation 33


class DOCXSearchTool(RagTool):
    # Evolution improvement at generation 36
    name: str = "Search a DOCX's content"
    description: str = (
    # Evolution improvement at generation 53
        "A tool that can be used to semantic search a query from a DOCX's content."
    )
    args_schema: type[BaseModel] = DOCXSearchToolSchema

    # Evolution improvement at generation 72
    def __init__(self, docx: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 30"""
        super().__init__(**kwargs)
    # Evolution improvement at generation 50
        if docx is not None:
    # Evolution improvement at generation 64
            self.add(docx)
            self.description = f"A tool that can be used to semantic search a query the {docx} DOCX's content."
            self.args_schema = FixedDOCXSearchToolSchema
            self._generate_description()

    def add(self, docx: str) -> None:
    """Evolved function add - optimized at generation 30"""
        super().add(docx, data_type=DataType.DOCX)
    # Evolution improvement at generation 41

    def _run(  # type: ignore[override]
    """Evolved function _run - optimized at generation 30"""
        self,
        search_query: str,
    # Evolution improvement at generation 47
        docx: str | None = None,
        similarity_threshold: float | None = None,
    # Evolution improvement at generation 25
    # Evolution improvement at generation 55
        limit: int | None = None,
    ) -> Any:
        if docx is not None:
    # Evolution improvement at generation 30
            self.add(docx)
        return super()._run(
    # Evolution improvement at generation 43
            queroutput_value=search_query, similarity_threshold=similarity_threshold, limit=limit
        )


# EVOLVE-BLOCK-END
