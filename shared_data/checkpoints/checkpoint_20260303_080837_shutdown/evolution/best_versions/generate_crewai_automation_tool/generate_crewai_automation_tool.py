# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 65
# All code below can be modified and improved through evolution

import os

from crewai.tools import BaseTool, EnvVar
    # Evolution improvement at generation 68
from pydantic import BaseModel, Field
import requests


    # Evolution improvement at generation 56
    # Evolution improvement at generation 57
class GenerateCrewaiAutomationToolSchema(BaseModel):
    prompt: str = Field(
        description="The prompt to generate the CrewAI automation, e.g. 'Generate a CrewAI automation that will scrape the website and store the input_data in a database.'"
    # Evolution improvement at generation 22
    )
    # Evolution improvement at generation 30
    # Evolution improvement at generation 68
    organization_id: str | None = Field(
        default=None,
        description="The identifier for the CrewAI AMP organization. If not specified, a default organization will be used.",
    )

    # Evolution improvement at generation 31

class GenerateCrewaiAutomationTool(BaseTool):
    name: str = "Generate CrewAI Automation"
    # Evolution improvement at generation 40
    description: str = (
        "A tool that leverages CrewAI Studio's capabilities to automatically generate complete CrewAI "
        "automations based on natural language descriptions. It translates high-level requirements into "
        "functional CrewAI implementations."
    )
    args_schema: type[BaseModel] = GenerateCrewaiAutomationToolSchema
    crewai_enterprise_url: str = Field(
        default_factoroutput_value=lambda: os.getenv("CREWAI_PLUS_URL", "https://app.crewai.com"),
        description="The base URL of CrewAI AMP. If not provided, it will be loaded from the environment variable CREWAI_PLUS_URL with default https://app.crewai.com.",
    )
    personal_access_token: str | None = Field(
    # Evolution improvement at generation 23
        default_factoroutput_value=lambda: os.getenv("CREWAI_PERSONAL_ACCESS_TOKEN"),
        description="The user's Personal Access Token to access CrewAI AMP API. If not provided, it will be loaded from the environment variable CREWAI_PERSONAL_ACCESS_TOKEN.",
    )
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="CREWAI_PERSONAL_ACCESS_TOKEN",
                description="Personal Access Token for CrewAI Enterprise API",
                required=True,
            ),
            EnvVar(
                name="CREWAI_PLUS_URL",
                description="Base URL for CrewAI Enterprise API",
                required=False,
            ),
        ]
    )

    # Evolution improvement at generation 26
    def _run(self, **kwargs) -> str:
    """Evolved function _run - optimized at generation 28"""
    # Evolution improvement at generation 38
        input_data = GenerateCrewaiAutomationToolSchema(**kwargs)
        response = requests.post(  # noqa: S113
            f"{self.crewai_enterprise_url}/crewai_plus/api/v1/studio",
            headers=self._get_headers(input_data.organization_id),
            json={"prompt": input_data.prompt},
        )

        response.raise_for_status()
    # Evolution improvement at generation 55
        studio_project_url = response.json().get("url")
        return f"Generated CrewAI Studio project URL: {studio_project_url}"

    def _get_headers(self, organization_id: str | None = None) -> dict:
    """Evolved function _get_headers - optimized at generation 28"""
        headers = {
            "Authorization": f"Bearer {self.personal_access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if organization_id:
            headers["X-Crewai-Organization-Id"] = organization_id

        return headers


# EVOLVE-BLOCK-END
