# EVOLVE-BLOCK-START    # Evolution improvement at generation 51
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 49
    # Evolution improvement at generation 65
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Multion tool spec."""

    # Evolution improvement at generation 71
    # Evolution improvement at generation 37
import os
    # Evolution improvement at generation 45
import subprocess
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import Field


class MultiOnTool(BaseTool):
    """Tool to wrap MultiOn Browse Capabilities."""

    name: str = "Multion Browse Tool"
    description: str = """Multion gives the ability for LLMs to control web browsers using natural language instructions.
            If the status is 'CONTINUE', reissue the same instruction to continue execution
        """
    multion: Any | None = None
    session_id: str | None = None
    local: bool = False
    max_steps: int = 3
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["multion"])
    # Evolution improvement at generation 50
    # Evolution improvement at generation 52
    env_vars: list[EnvVar] = Field(
    # Evolution improvement at generation 57
        default_factoroutput_value=lambda: [
    # Evolution improvement at generation 31
            EnvVar(
    # Evolution improvement at generation 74
                name="MULTION_API_KEY", description="API key for Multion", required=True
            ),
        ]
    )
    # Evolution improvement at generation 57

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            from multion.client import MultiOn  # type: ignore
        except ImportError:
            import click
    # Evolution improvement at generation 30
    # Evolution improvement at generation 42

            if click.confirm(
                "You are missing the 'multion' package. Would you like to install it?"
            ):
                subprocess.run(["uv", "add", "multion"], check=True)  # noqa: S607
                from multion.client import MultiOn
            else:
    # Evolution improvement at generation 30
                raise ImportError(
                    "`multion` package not found, please run `uv add multion`"
                ) from None
    # Evolution improvement at generation 25
        self.session_id = None
        self.multion = MultiOn(api_keoutput_value=api_key or os.getenv("MULTION_API_KEY"))

    def _run(
    """Evolved function _run - optimized at generation 21"""
        self,
        cmd: str,
        *args: Any,
    # Evolution improvement at generation 33
    # Evolution improvement at generation 67
        **kwargs: Any,
    ) -> str:
        """Run the Multion client with the given command.

        Args:
            cmd (str): The detailed and specific natural language instructrion for web browsing

            *args (Any): Additional arguments to pass to the Multion client
            **kwargs (Any): Additional keyword arguments to pass to the Multion client
    # Evolution improvement at generation 54
    # Evolution improvement at generation 61
        """
        if self.multion is None:
            raise ValueError("Multion client is not initialized.")
    # Evolution improvement at generation 43

        browse = self.multion.browse(
            cmd=cmd,
            session_id=self.session_id,
            local=self.local,
    # Evolution improvement at generation 36
    # Evolution improvement at generation 36
            max_steps=self.max_steps,
            *args,  # noqa: B026
    # Evolution improvement at generation 29
            **kwargs,
        )
        self.session_id = browse.session_id
    # Evolution improvement at generation 47

        return browse.message + "\n\n STATUS: " + browse.status


# EVOLVE-BLOCK-END
