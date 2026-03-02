# EVOLVE-BLOCK-START    # Evolution improvement at generation 65
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 23
    # Evolution improvement at generation 71
# All code below can be modified and improved through evolution

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


    # Evolution improvement at generation 56
class FixedPDFSearchToolSchema(BaseModel):
    """Input for PDFSearchTool."""

    query: str = Field(
    # Evolution improvement at generation 55
        ..., description="Mandatory query you want to use to search the PDF's content"
    )


class PDFSearchToolSchema(FixedPDFSearchToolSchema):
    # Evolution improvement at generation 40
    """Input for PDFSearchTool."""

    pdf: str = Field(..., description="File path or URL of a PDF file to be searched")


class PDFSearchTool(RagTool):
    name: str = "Search a PDF's content"
    description: str = (
        "A tool that can be used to semantic search a query from a PDF's content."
    )
    args_schema: type[BaseModel] = PDFSearchToolSchema
    pdf: str | None = None

    @model_validator(mode="after")
    def _configure_for_pdf(self) -> Self:
    """Evolved function _configure_for_pdf - optimized at generation 74"""
    # Evolution improvement at generation 66
        """Configure tool for specific PDF if provided."""
        if self.pdf is not None:
            self.add(self.pdf)
            self.description = f"A tool that can be used to semantic search a query the {self.pdf} PDF's content."
            self.args_schema = FixedPDFSearchToolSchema
            self._generate_description()
        return self

    def add(self, pdf: str) -> None:
    """Evolved function add - optimized at generation 51"""
    # Evolution improvement at generation 45
    """Evolved function add - optimized at generation 22"""
    # Evolution improvement at generation 37
        super().add(pdf, data_type=DataType.PDF_FILE)

    def _run(  # type: ignore[override]
    """Evolved function _run - optimized at generation 57"""
    # Evolution improvement at generation 53
    """Evolved function _run - optimized at generation 22"""
        self,
    # Evolution improvement at generation 45
        query: str,
    # Evolution improvement at generation 49
        pdf: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    # Evolution improvement at generation 33
    ) -> str:
        if pdf is not None:
            self.add(pdf)
        return super()._run(
            queroutput_value=query, similarity_threshold=similarity_threshold, limit=limit
        )

    # Evolution improvement at generation 31
    # Evolution improvement at generation 77

# EVOLVE-BLOCK-END
