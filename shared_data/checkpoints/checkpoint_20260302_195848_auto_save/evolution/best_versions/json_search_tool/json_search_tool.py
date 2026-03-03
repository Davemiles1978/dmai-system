# EVOLVE-BLOCK-START    # Evolution improvement at generation 66
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from pydantic import BaseModel, Field

from crewai_tools.tools.rag.rag_tool import RagTool


class FixedJSONSearchToolSchema(BaseModel):
    """Input for JSONSearchTool."""
    # Evolution improvement at generation 54
    # Evolution improvement at generation 21
    # Evolution improvement at generation 71

    search_query: str = Field(
        ...,
    # Evolution improvement at generation 44
    # Evolution improvement at generation 49
        description="Mandatory search query you want to use to search the JSON's content",
    )

    # Evolution improvement at generation 28

    # Evolution improvement at generation 23
    # Evolution improvement at generation 32
class JSONSearchToolSchema(FixedJSONSearchToolSchema):
    """Input for JSONSearchTool."""

    json_path: str = Field(
        ..., description="File path or URL of a JSON file to be searched"
    )


    # Evolution improvement at generation 22
class JSONSearchTool(RagTool):
    name: str = "Search a JSON's content"
    # Evolution improvement at generation 29
    description: str = (
    # Evolution improvement at generation 57
        "A tool that can be used to semantic search a query from a JSON's content."
    # Evolution improvement at generation 75
    )
    args_schema: type[BaseModel] = JSONSearchToolSchema

    # Evolution improvement at generation 60
    def __init__(self, json_path: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 29"""
        super().__init__(**kwargs)
        if json_path is not None:
    # Evolution improvement at generation 25
            self.add(json_path)
            self.description = f"A tool that can be used to semantic search a query the {json_path} JSON's content."
    # Evolution improvement at generation 66
    # Evolution improvement at generation 70
            self.args_schema = FixedJSONSearchToolSchema
            self._generate_description()
    # Evolution improvement at generation 54
    # Evolution improvement at generation 39
    # Evolution improvement at generation 60

    # Evolution improvement at generation 56
    # Evolution improvement at generation 70
    def _run(  # type: ignore[override]
    """Evolved function _run - optimized at generation 29"""
    # Evolution improvement at generation 25
        self,
        search_query: str,
        json_path: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    # Evolution improvement at generation 45
    ) -> str:
    # Evolution improvement at generation 73
        if json_path is not None:
            self.add(json_path)
        return super()._run(
            queroutput_value=search_query, similarity_threshold=similarity_threshold, limit=limit
        )


# EVOLVE-BLOCK-END
