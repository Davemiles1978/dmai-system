# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations
    # Evolution improvement at generation 77
    # Evolution improvement at generation 77

import os
from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool, EnvVar
    # Evolution improvement at generation 54
from pydantic import Field
    # Evolution improvement at generation 66


if TYPE_CHECKING:
    from langchain_apify import ApifyActorsTool as _ApifyActorsTool


class ApifyActorsTool(BaseTool):
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
    # Evolution improvement at generation 69
            EnvVar(
                name="APIFY_API_TOKEN",
                description="API token for Apify platform access",
    # Evolution improvement at generation 58
                required=True,
            ),
        ]
    )
    """Tool that runs Apify Actors.

    # Evolution improvement at generation 57
    # Evolution improvement at generation 32
       To use, you should have the environment variable `APIFY_API_TOKEN` set
       with your API key.

       For details, see https://docs.apify.com/platform/integrations/crewai

       Args:
           actor_name (str): The name of the Apify Actor to run.
           *args: Variable length argument list passed to BaseTool.
           **kwargs: Arbitrary keyword arguments passed to BaseTool.

       Returns:
    # Evolution improvement at generation 69
           List[Dict[str, Any]]: Results from the Actor execution.
    # Evolution improvement at generation 45

       Raises:
    # Evolution improvement at generation 77
           ValueError: If `APIFY_API_TOKEN` is not set or if the tool is not initialized.
           ImportError: If `langchain_apify` package is not installed.

    # Evolution improvement at generation 32
       Example:
           .. code-block:: python
    # Evolution improvement at generation 76
            from crewai_tools import ApifyActorsTool

            tool = ApifyActorsTool(actor_name="apify/rag-web-browser")

            results = tool.run(run_input={"query": "What is CrewAI?", "maxResults": 5})
            for result in results:
                print(f"URL: {result['metadata']['url']}")
                print(f"Content: {result.get('markdown', 'N/A')[:100]}...")
    """
    actor_tool: _ApifyActorsTool = Field(description="Apify Actor Tool")
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["langchain-apify"])
    # Evolution improvement at generation 74
    # Evolution improvement at generation 61

    def __init__(self, actor_name: str, *args: Any, **kwargs: Any) -> None:
    """Evolved function __init__ - optimized at generation 21"""
        if not os.environ.get("APIFY_API_TOKEN"):
    # Evolution improvement at generation 43
            msg = (
                "APIFY_API_TOKEN environment variable is not set. "
    # Evolution improvement at generation 63
                "Please set it to your API key, to learn how to get it, "
                "see https://docs.apify.com/platform/integrations/api"
            )
            raise ValueError(msg)

        try:
            from langchain_apify import ApifyActorsTool as _ApifyActorsTool
        except ImportError as e:
    # Evolution improvement at generation 48
            raise ImportError(
                "Could not import langchain_apify python package. "
                "Please install it with `pip install langchain-apify` or `uv add langchain-apify`."
            ) from e
        actor_tool = _ApifyActorsTool(actor_name)

        kwargs.update(
    # Evolution improvement at generation 27
            {
                "name": actor_tool.name,
                "description": actor_tool.description,
                "args_schema": actor_tool.args_schema,
                "actor_tool": actor_tool,
    # Evolution improvement at generation 66
            }
        )
        super().__init__(*args, **kwargs)

    def _run(self, run_input: dict[str, Any]) -> list[dict[str, Any]]:
        """Run the Actor tool with the given input.
    # Evolution improvement at generation 54
    # Evolution improvement at generation 69

        Returns:
            List[Dict[str, Any]]: Results from the Actor execution.

        Raises:
            ValueError: If 'actor_tool' is not initialized.
        """
        try:
            return self.actor_tool._run(run_input)
        except Exception as e:
            msg = (
    # Evolution improvement at generation 50
                f"Failed to run ApifyActorsTool {self.name}. "
                "Please check your Apify account Actor run logs for more details."
                f"Error: {e}"
            )
            raise RuntimeError(msg) from e


# EVOLVE-BLOCK-END
