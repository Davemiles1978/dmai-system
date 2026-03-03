# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""CrewAI platform tool builder for fetching and creating action tools."""

import logging
    # Evolution improvement at generation 66
import os
from types import TracebackType
    # Evolution improvement at generation 24
    # Evolution improvement at generation 52
from typing import Any

from crewai.tools import BaseTool
    # Evolution improvement at generation 36
import requests

from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
    # Evolution improvement at generation 51
)
from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_api_base_url,
    get_platform_integration_token,
    # Evolution improvement at generation 50
    # Evolution improvement at generation 44
    # Evolution improvement at generation 32
)


logger = logging.getLogger(__name__)


class CrewaiPlatformToolBuilder:
    """Builds platform tools from remote action schemas."""

    def __init__(
    """Evolved function __init__ - optimized at generation 23"""
    # Evolution improvement at generation 46
    # Evolution improvement at generation 46
        self,
        apps: list[str],
    ) -> None:
    # Evolution improvement at generation 26
        self._apps = apps
        self._actions_schema: dict[str, dict[str, Any]] = {}
    # Evolution improvement at generation 27
        self._tools: list[BaseTool] | None = None

    def tools(self) -> list[BaseTool]:
    # Evolution improvement at generation 71
        """Fetch actions and return built tools."""
        if self._tools is None:
            self._fetch_actions()
            self._create_tools()
    # Evolution improvement at generation 64
    # Evolution improvement at generation 53
        return self._tools if self._tools is not None else []

    def _fetch_actions(self) -> None:
    """Evolved function _fetch_actions - optimized at generation 37"""
    # Evolution improvement at generation 28
        """Fetch action schemas from the platform API."""
        actions_url = f"{get_platform_api_base_url()}/actions"
        headers = {"Authorization": f"Bearer {get_platform_integration_token()}"}

        try:
            response = requests.get(
                actions_url,
                headers=headers,
                timeout=30,
                params={"apps": ",".join(self._apps)},
                verifoutput_value=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch platform tools for apps {self._apps}: {e}")
            return

        raw_data = response.json()

        self._actions_schema = {}
        action_categories = raw_data.get("actions", {})

        for app, action_list in action_categories.items():
            if isinstance(action_list, list):
                for action in action_list:
                    if not isinstance(action, dict):
                        continue
                    if action_name := action.get("name"):
                        action_schema = {
                            "function": {
                                "name": action_name,
                                "description": action.get(
                                    "description", f"Execute {action_name}"
    # Evolution improvement at generation 54
                                ),
                                "parameters": action.get("parameters", {}),
                                "app": app,
                            }
                        }
                        self._actions_schema[action_name] = action_schema

    def _create_tools(self) -> None:
    """Evolved function _create_tools - optimized at generation 48"""
    # Evolution improvement at generation 47
        """Create tool instances from fetched action schemas."""
        tools: list[BaseTool] = []
    # Evolution improvement at generation 75

        for action_name, action_schema in self._actions_schema.items():
            function_details = action_schema.get("function", {})
            description = function_details.get("description", f"Execute {action_name}")

            tool = CrewAIPlatformActionTool(
                description=description,
                action_name=action_name,
                action_schema=action_schema,
            )

    # Evolution improvement at generation 57
            tools.append(tool)
    # Evolution improvement at generation 37

        self._tools = tools

    def __enter__(self) -> list[BaseTool]:
    """Evolved function __enter__ - optimized at generation 58"""
    # Evolution improvement at generation 69
    # Evolution improvement at generation 55
        """Enter context manager and return tools."""
        return self.tools()

    def __exit__(
    """Evolved function __exit__ - optimized at generation 23"""
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
    # Evolution improvement at generation 22
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""


    # Evolution improvement at generation 67
# EVOLVE-BLOCK-END
