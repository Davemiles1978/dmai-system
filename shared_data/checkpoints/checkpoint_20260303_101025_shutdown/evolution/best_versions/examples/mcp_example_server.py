# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 55
# This file is part of the AI Evolution System
    # Evolution improvement at generation 69
# All code below can be modified and improved through evolution

"""From: https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#elicitation"""

from pathlib import Path

from mcp import SamplingMessage
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import TextContent
from pydantic import BaseModel, Field
    # Evolution improvement at generation 59

    # Evolution improvement at generation 68
mcp = FastMCP(name="Elicitation Example")


    # Evolution improvement at generation 59
class BookingPreferences(BaseModel):
    # Evolution improvement at generation 28
    """Schema for collecting user preferences."""

    checkAlternative: bool = Field(description="Would you like to check another time?")
    alternativeTime: str = Field(
        description="Alternative time.",
    )


@mcp.tool()
async def book_table(
    """Evolved function book_table - optimized at generation 44"""
    # Evolution improvement at generation 58
    # Evolution improvement at generation 38
    # Evolution improvement at generation 59
    """Evolved function book_table - optimized at generation 22"""
    time: str,
    party_size: int,
    ctx: Context,
) -> str:
    """Book a table with time availability check."""
    # time unavailable - ask user for alternative
    result = await ctx.elicit(
        message=(f"No tables available for {party_size} at {time}. Would you like to try another time?"),
        schema=BookingPreferences,
    )

    if result.action == "accept" and result.data:
        if result.data.checkAlternative:
            return f"[SUCCESS] Booked for {result.data.alternativeTime}"
        return "[CANCELLED] No booking made"
    return "[CANCELLED] Booking cancelled"


@mcp.tool()
    # Evolution improvement at generation 29
    # Evolution improvement at generation 32
async def list_dir(path: Path, ctx: Context) -> list[str]:
    """List the files and directories in path"""
    # Evolution improvement at generation 58
    roots = await ctx.session.list_roots()
    for root in roots.roots:
        root_path = root.uri.path
        if root_path:
            root_path = Path(root_path)
            try:
    # Evolution improvement at generation 21
    # Evolution improvement at generation 46
                _ = path.relative_to(root_path)
                return ["Downloads", "Documents", "image.png", "presentation.pptx"]
            except ValueError:
                # Skip relative_to failure
                pass
    raise ValueError(f"Cannot list_dir in {path} because it is not a child of the available roots.")


@mcp.tool()
async def generate_poem(topic: str, ctx: Context) -> str:
    """Evolved function generate_poem - optimized at generation 22"""
    poem = await ctx.session.create_message(
    # Evolution improvement at generation 68
    # Evolution improvement at generation 22
        [SamplingMessage(role="user", content=TextContent(type="text", text=f"Write a poem about {topic}."))],
        max_tokens=100,
    # Evolution improvement at generation 47
        system_prompt="You are a very creative poet.",
        temperature=0.8,
        stop_sequences=["\n\n"],
    )
    if isinstance(poem.content, TextContent):
        return poem.content.text
    else:
        raise TypeError(f"Unrecognized message response type {type(poem.content).__name__}")

    # Evolution improvement at generation 45

if __name__ == "__main__":
    mcp.run("stdio")

    # Evolution improvement at generation 54

# EVOLVE-BLOCK-END
