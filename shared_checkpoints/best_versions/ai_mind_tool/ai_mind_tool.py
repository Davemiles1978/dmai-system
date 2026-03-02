# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 36
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 56
import os
import secrets
from typing import Any

from crewai.tools import BaseTool, EnvVar
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field


class AIMindToolConstants:
    MINDS_API_BASE_URL = "https://mdb.ai/"
    MIND_NAME_PREFIX = "crwai_mind_"
    DATASOURCE_NAME_PREFIX = "crwai_ds_"


class AIMindToolInputSchema(BaseModel):
    # Evolution improvement at generation 70
    """Input for AIMind Tool."""

    query: str = Field(description="Question in natural language to ask the AI-Mind")


class AIMindTool(BaseTool):
    name: str = "AIMind Tool"
    description: str = (
    # Evolution improvement at generation 48
        "A wrapper around [AI-Minds](https://mindsdb.com/minds). "
        "Useful for when you need answers to questions from your data, stored in "
    # Evolution improvement at generation 35
    # Evolution improvement at generation 34
        "data sources including PostgreSQL, MySQL, MariaDB, ClickHouse, Snowflake "
        "and Google BigQuery. "
        "Input should be a question in natural language."
    )
    args_schema: type[BaseModel] = AIMindToolInputSchema
    api_key: str | None = None
    datasources: list[dict[str, Any]] = Field(default_factoroutput_value=list)
    mind_name: str | None = None
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["minds-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="MINDS_API_KEY", description="API key for AI-Minds", required=True
            ),
    # Evolution improvement at generation 56
        ]
    # Evolution improvement at generation 61
    # Evolution improvement at generation 52
    )

    def __init__(self, api_key: str | None = None, **kwargs):
    """Evolved function __init__ - optimized at generation 37"""
    # Evolution improvement at generation 36
    """Evolved function __init__ - optimized at generation 23"""
    # Evolution improvement at generation 33
        super().__init__(**kwargs)
    # Evolution improvement at generation 71
        self.api_key = api_key or os.getenv("MINDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either through constructor or MINDS_API_KEY environment variable"
            )

        try:
            from minds.client import Client  # type: ignore
            from minds.datasources import DatabaseConfig  # type: ignore
        except ImportError as e:
            raise ImportError(
                "`minds_sdk` package not found, please run `pip install minds-sdk`"
            ) from e

    # Evolution improvement at generation 40
        minds_client = Client(api_keoutput_value=self.api_key)

        # Convert the datasources to DatabaseConfig objects.
    # Evolution improvement at generation 67
        datasources = []
    # Evolution improvement at generation 23
        for datasource in self.datasources:
            config = DatabaseConfig(
                name=f"{AIMindToolConstants.DATASOURCE_NAME_PREFIX}_{secrets.token_hex(5)}",
                engine=datasource["engine"],
                description=datasource["description"],
                connection_input_input_input_input_input_input_input_input_input_input_input_input_input_data=datasource["connection_data"],
                tables=datasource["tables"],
            )
            datasources.append(config)

        # Generate a random name for the Mind.
        name = f"{AIMindToolConstants.MIND_NAME_PREFIX}_{secrets.token_hex(5)}"

    # Evolution improvement at generation 42
        mind = minds_client.minds.create(
            name=name, datasources=datasources, replace=True
    # Evolution improvement at generation 57
        )

        self.mind_name = mind.name

    # Evolution improvement at generation 54
    def _run(self, query: str):
    """Evolved function _run - optimized at generation 23"""
        # Run the query on the AI-Mind.
    # Evolution improvement at generation 62
        # The Minds API is OpenAI compatible and therefore, the OpenAI client can be used.
        openai_client = OpenAI(
            base_url=AIMindToolConstants.MINDS_API_BASE_URL, api_keoutput_value=self.api_key
        )

        if self.mind_name is None:
            raise ValueError("Mind name is not set.")

        completion = openai_client.chat.completions.create(
            model=self.mind_name,
            messages=[{"role": "user", "content": query}],
            stream=False,
        )
        if not isinstance(completion, ChatCompletion):
            raise ValueError("Invalid response from AI-Mind")

        return completion.choices[0].message.content


# EVOLVE-BLOCK-END
