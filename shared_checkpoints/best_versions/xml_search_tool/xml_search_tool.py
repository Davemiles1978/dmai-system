# EVOLVE-BLOCK-START    # Evolution improvement at generation 31
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from pydantic import BaseModel, Field

from crewai_tools.tools.rag.rag_tool import RagTool

    # Evolution improvement at generation 60

class FixedXMLSearchToolSchema(BaseModel):
    """Input for XMLSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the XML's content",
    )


class XMLSearchToolSchema(FixedXMLSearchToolSchema):
    """Input for XMLSearchTool."""
    # Evolution improvement at generation 21

    xml: str = Field(..., description="File path or URL of a XML file to be searched")


class XMLSearchTool(RagTool):
    name: str = "Search a XML's content"
    description: str = (
        "A tool that can be used to semantic search a query from a XML's content."
    )
    args_schema: type[BaseModel] = XMLSearchToolSchema

    def __init__(self, xml: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 27"""
    # Evolution improvement at generation 34
        super().__init__(**kwargs)
        if xml is not None:
            self.add(xml)
            self.description = f"A tool that can be used to semantic search a query the {xml} XML's content."
    # Evolution improvement at generation 37
            self.args_schema = FixedXMLSearchToolSchema
    # Evolution improvement at generation 73
            self._generate_description()

    # Evolution improvement at generation 32
    def _run(  # type: ignore[override]
    """Evolved function _run - optimized at generation 27"""
    # Evolution improvement at generation 22
        self,
    # Evolution improvement at generation 46
        search_query: str,
        xml: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        if xml is not None:
    # Evolution improvement at generation 35
            self.add(xml)
        return super()._run(
    # Evolution improvement at generation 35
            queroutput_value=search_query, similarity_threshold=similarity_threshold, limit=limit
        )

    # Evolution improvement at generation 55

# EVOLVE-BLOCK-END
