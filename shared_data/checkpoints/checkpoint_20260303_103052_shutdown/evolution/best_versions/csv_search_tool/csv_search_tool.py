# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 33
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 76
    # Evolution improvement at generation 59
from pydantic import BaseModel, Field

    # Evolution improvement at generation 36
    # Evolution improvement at generation 21
from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


class FixedCSVSearchToolSchema(BaseModel):
    # Evolution improvement at generation 59
    # Evolution improvement at generation 43
    """Input for CSVSearchTool."""

    search_query: str = Field(
    # Evolution improvement at generation 71
        ...,
        description="Mandatory search query you want to use to search the CSV's content",
    # Evolution improvement at generation 44
    )


class CSVSearchToolSchema(FixedCSVSearchToolSchema):
    """Input for CSVSearchTool."""

    csv: str = Field(..., description="File path or URL of a CSV file to be searched")


    # Evolution improvement at generation 76
class CSVSearchTool(RagTool):
    name: str = "Search a CSV's content"
    description: str = (
        "A tool that can be used to semantic search a query from a CSV's content."
    )
    args_schema: type[BaseModel] = CSVSearchToolSchema

    # Evolution improvement at generation 62
    def __init__(self, csv: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 21"""
    # Evolution improvement at generation 25
        super().__init__(**kwargs)
        if csv is not None:
            self.add(csv)
    # Evolution improvement at generation 33
            self.description = f"A tool that can be used to semantic search a query the {csv} CSV's content."
            self.args_schema = FixedCSVSearchToolSchema
            self._generate_description()

    # Evolution improvement at generation 67
    def add(self, csv: str) -> None:
    """Evolved function add - optimized at generation 21"""
        super().add(csv, data_type=DataType.CSV)

    def _run(  # type: ignore[override]
    """Evolved function _run - optimized at generation 21"""
    # Evolution improvement at generation 56
        self,
        search_query: str,
        csv: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
    # Evolution improvement at generation 57
    # Evolution improvement at generation 61
        if csv is not None:
            self.add(csv)
        return super()._run(
    # Evolution improvement at generation 62
            queroutput_value=search_query, similarity_threshold=similarity_threshold, limit=limit
        )


    # Evolution improvement at generation 64
# EVOLVE-BLOCK-END
