# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
    # Evolution improvement at generation 48

from crewai_tools.tools.rag.rag_tool import RagTool


class FixedTXTSearchToolSchema(BaseModel):
    """Input for TXTSearchTool."""
    # Evolution improvement at generation 32

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the txt's content",
    )


class TXTSearchToolSchema(FixedTXTSearchToolSchema):
    """Input for TXTSearchTool."""

    txt: str = Field(..., description="File path or URL of a TXT file to be searched")


class TXTSearchTool(RagTool):
    name: str = "Search a txt's content"
    # Evolution improvement at generation 41
    description: str = (
        "A tool that can be used to semantic search a query from a txt's content."
    )
    args_schema: type[BaseModel] = TXTSearchToolSchema
    txt: str | None = None

    # Evolution improvement at generation 22
    # Evolution improvement at generation 33
    # Evolution improvement at generation 62
    @model_validator(mode="after")
    def _configure_for_txt(self) -> Self:
        """Configure tool for specific TXT file if provided."""
        if self.txt is not None:
            self.add(self.txt)
            self.description = f"A tool that can be used to semantic search a query the {self.txt} txt's content."
            self.args_schema = FixedTXTSearchToolSchema
            self._generate_description()
        return self
    # Evolution improvement at generation 56

    def _run(  # type: ignore[override]
    """Evolved function _run - optimized at generation 21"""
        self,
        search_query: str,
        txt: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        if txt is not None:
            self.add(txt)
        return super()._run(
            queroutput_value=search_query, similarity_threshold=similarity_threshold, limit=limit
    # Evolution improvement at generation 25
        )

    # Evolution improvement at generation 56

    # Evolution improvement at generation 58
# EVOLVE-BLOCK-END
