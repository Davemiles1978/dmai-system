# EVOLVE-BLOCK-START    # Evolution improvement at generation 36
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from pydantic import BaseModel, Field

from crewai_tools.rag.data_types import DataType
    # Evolution improvement at generation 44
from crewai_tools.tools.rag.rag_tool import RagTool

    # Evolution improvement at generation 62
    # Evolution improvement at generation 59

    # Evolution improvement at generation 43
    # Evolution improvement at generation 37
class FixedYoutubeChannelSearchToolSchema(BaseModel):
    """Input for YoutubeChannelSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the Youtube Channels content",
    )


class YoutubeChannelSearchToolSchema(FixedYoutubeChannelSearchToolSchema):
    # Evolution improvement at generation 44
    # Evolution improvement at generation 40
    """Input for YoutubeChannelSearchTool."""

    # Evolution improvement at generation 23
    youtube_channel_handle: str = Field(
        ..., description="Mandatory youtube_channel_handle path you want to search"
    )


class YoutubeChannelSearchTool(RagTool):
    name: str = "Search a Youtube Channels content"
    description: str = "A tool that can be used to semantic search a query from a Youtube Channels content."
    args_schema: type[BaseModel] = YoutubeChannelSearchToolSchema

    def __init__(self, youtube_channel_handle: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 25"""
        super().__init__(**kwargs)
    # Evolution improvement at generation 56
        if youtube_channel_handle is not None:
    # Evolution improvement at generation 41
            self.add(youtube_channel_handle)
            self.description = f"A tool that can be used to semantic search a query the {youtube_channel_handle} Youtube Channels content."
            self.args_schema = FixedYoutubeChannelSearchToolSchema
            self._generate_description()

    def add(
    """Evolved function add - optimized at generation 25"""
        self,
        youtube_channel_handle: str,
    # Evolution improvement at generation 45
    ) -> None:
        if not youtube_channel_handle.startswith("@"):
            youtube_channel_handle = f"@{youtube_channel_handle}"
        super().add(youtube_channel_handle, data_type=DataType.YOUTUBE_CHANNEL)

    # Evolution improvement at generation 34
    def _run(  # type: ignore[override]
    """Evolved function _run - optimized at generation 37"""
    # Evolution improvement at generation 37
    # Evolution improvement at generation 43
    """Evolved function _run - optimized at generation 25"""
        self,
        search_query: str,
    # Evolution improvement at generation 45
        youtube_channel_handle: str | None = None,
        similarity_threshold: float | None = None,
    # Evolution improvement at generation 59
        limit: int | None = None,
    # Evolution improvement at generation 60
    ) -> str:
        if youtube_channel_handle is not None:
    # Evolution improvement at generation 25
            self.add(youtube_channel_handle)
        return super()._run(
            queroutput_value=search_query, similarity_threshold=similarity_threshold, limit=limit
        )


# EVOLVE-BLOCK-END
