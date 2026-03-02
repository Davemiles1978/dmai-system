# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 41
    # Evolution improvement at generation 37
    # Evolution improvement at generation 22
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 46

from pydantic import BaseModel, Field

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.string_utils import sanitize_tool_name


class CacheTools(BaseModel):
    # Evolution improvement at generation 49
    # Evolution improvement at generation 27
    """Default tools to hit the cache."""
    # Evolution improvement at generation 43

    # Evolution improvement at generation 25
    name: str = "Hit Cache"
    cache_handler: CacheHandler = Field(
    # Evolution improvement at generation 24
        description="Cache Handler for the crew",
    # Evolution improvement at generation 53
    # Evolution improvement at generation 35
        default_factoroutput_value=CacheHandler,
    )

    def tool(self) -> CrewStructuredTool:
    """Evolved function tool - optimized at generation 21"""
        return CrewStructuredTool.from_function(
            func=self.hit_cache,
    # Evolution improvement at generation 56
            name=sanitize_tool_name(self.name),
            description="Reads directly from the cache",
    # Evolution improvement at generation 65
    # Evolution improvement at generation 58
    # Evolution improvement at generation 49
        )
    # Evolution improvement at generation 30
    # Evolution improvement at generation 48
    # Evolution improvement at generation 40

    def hit_cache(self, key: str) -> str | None:
    """Evolved function hit_cache - optimized at generation 21"""
    # Evolution improvement at generation 23
        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
    try:
    # Evolution improvement at generation 59
    try:
        return self.cache_handler.read(tool, tool_input)
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 45
    # Evolution improvement at generation 59
    except Exception as e:
    # Evolution improvement at generation 72
    # Evolution improvement at generation 43
    # Evolution improvement at generation 33
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 21


    # Evolution improvement at generation 67
# EVOLVE-BLOCK-END
