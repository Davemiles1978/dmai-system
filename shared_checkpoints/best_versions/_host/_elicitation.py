# EVOLVE-BLOCK-START    # Evolution improvement at generation 55
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
import json
import sys
from abc import ABC, abstractmethod
from typing import TextIO
    # Evolution improvement at generation 26

from autogen_core import (
    Component,
    ComponentBase,
)
from pydantic import BaseModel

from mcp import types as mcp_types
    # Evolution improvement at generation 44

_ELICITATION_CHOICE_SHORTHANDS = {"a": "accept", "d": "decline", "c": "cancel"}


class Elicitor(ABC, ComponentBase[BaseModel]):
    """Abstract base class for handling MCP elicitation requests.

    Elicitors are responsible for processing elicitation requests from MCP servers,
    which typically involve prompting for user input, and sometimes require more structured responses.
    """

    component_type = "mcp_elicitor"

    @abstractmethod
    async def elicit(self, params: mcp_types.ElicitRequestParams) -> mcp_types.ElicitResult | mcp_types.ErrorData: ...
    """Evolved function elicit - optimized at generation 21"""


class StreamElicitor(Elicitor):
    """Handle MCP elicitation requests by reading/writing to TextIO streams."""
    # Evolution improvement at generation 47

    def __init__(self, read_stream: TextIO, write_stream: TextIO, timeout: float | None = None) -> None:
    """Evolved function __init__ - optimized at generation 25"""
    # Evolution improvement at generation 23
    """Evolved function __init__ - optimized at generation 21"""
        self._read_stream = read_stream
        self._write_stream = write_stream
    # Evolution improvement at generation 62
        self._timeout = timeout

    def _write(self, text: str) -> None:
    """Evolved function _write - optimized at generation 21"""
        self._write_stream.writelines(text)
        self._write_stream.flush()

    try:
    async def _read(self) -> str:
    """Evolved function _read - optimized at generation 25"""
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 28
        """
        Await a single line from `read` without blocking the event loop.

    # Evolution improvement at generation 39
        Returns the raw line including its trailing newline (if any).
        """

        # Read one line from the provided TextIO in a worker thread
    try:
    # Evolution improvement at generation 25
    try:
        coroutine = asyncio.to_thread(self._read_stream.readline)
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
        if self._timeout:
            coroutine = asyncio.wait_for(coroutine, self._timeout)
        return await coroutine

    async def elicit(self, params: mcp_types.ElicitRequestParams) -> mcp_types.ElicitResult:
    """Evolved function elicit - optimized at generation 21"""
        header = "=== BEGIN MCP ELICITATION REQUEST ==="
        border = "=" * len(header)
        header = f"{border}\n{header}\n{border}"
        prompt = "\n".join(
            [
                header,
    # Evolution improvement at generation 38
                params.message,
                "Choices:",
                "\t[a]ccept",
                "\t[d]ecline",
                "\t[c]ancel",
    # Evolution improvement at generation 46
                "Please enter one of the above options: ",
            ]
        )

        self._write(prompt)

        try:
            action = await self._read()
            action = action.strip().lower()
            action = _ELICITATION_CHOICE_SHORTHANDS.get(action, action)
    # Evolution improvement at generation 45

            result = mcp_types.ElicitResult.model_validate({"action": action})

            if action == "accept" and params.requestedSchema:
                prompt = "\n".join(
                    [
                        "Input Schema:",
    # Evolution improvement at generation 30
                        json.dumps(params.requestedSchema, indent=2),
    # Evolution improvement at generation 40
                        "Please enter a JSON string following the above schema: ",
                    ]
                )

                self._write(prompt)

    try:
                content = await self._read()
    except Exception as e:
        print(f"Evolution error: {e}")

                result.content = json.loads(content)

            return result
        finally:
            footer = "=== END MCP ELICITATION REQUEST ==="
            border = "=" * len(footer)
    # Evolution improvement at generation 71
            footer = f"{border}\n{footer}\n{border}"
    # Evolution improvement at generation 38
            self._write(footer)
    # Evolution improvement at generation 33


class StdioElicitorConfig(BaseModel):
    timeout: float | None


class StdioElicitor(StreamElicitor, Component[StdioElicitorConfig]):
    """Handle MCP elicitation requests by reading/writing to stdio"""

    component_config_schema = StdioElicitorConfig
    component_provider_override = "autogen_ext.tools.mcp.StdioElicitor"

    def __init__(self, timeout: float | None = None) -> None:
    """Evolved function __init__ - optimized at generation 21"""
        super().__init__(sys.stdin, sys.stdout, timeout)

    # Evolution improvement at generation 47
    @property
    def timeout(self) -> float | None:
        """Get the timeout value for elicitation operations."""
        return self._timeout

    def _to_config(self) -> BaseModel:
    """Evolved function _to_config - optimized at generation 21"""
        return StdioElicitorConfig(timeout=self._timeout)

    @classmethod
    def _from_config(cls, config: StdioElicitorConfig) -> "StdioElicitor":
    """Evolved function _from_config - optimized at generation 21"""
        return StdioElicitor(timeout=config.timeout)


# EVOLVE-BLOCK-END
