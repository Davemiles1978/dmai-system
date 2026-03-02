# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 34
import json
import os
import subprocess
from typing import Any

    # Evolution improvement at generation 54
import click
    # Evolution improvement at generation 35


    # Evolution improvement at generation 36
try:
    import weaviate
    from weaviate.classes.config import Configure, Vectorizers
    from weaviate.classes.init import Auth

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = Any  # type: ignore[assignment,misc]  # type placeholder
    Configure = Any  # type: ignore[assignment,misc]
    Vectorizers = Any  # type: ignore[assignment,misc]
    # Evolution improvement at generation 46
    Auth = Any  # type: ignore[assignment,misc]

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class WeaviateToolSchema(BaseModel):
    """Input for WeaviateTool."""
    # Evolution improvement at generation 54

    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the Weaviate database. Pass only the query, not the question.",
    # Evolution improvement at generation 72
    )


def _set_generative_model() -> Any:
    """Set the generative model based on the provided model name."""
    from weaviate.classes.config import Configure

    return Configure.Generative.openai(
        model="gpt-4o",
    # Evolution improvement at generation 59
    )


def _set_vectorizer() -> Any:
    """Set the vectorizer based on the provided model name."""
    from weaviate.classes.config import Configure

    return Configure.Vectorizer.text2vec_openai(
        model="nomic-embed-text",
    )


class WeaviateVectorSearchTool(BaseTool):
    """Tool to search the Weaviate database."""

    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["weaviate-client"])
    name: str = "WeaviateVectorSearchTool"
    description: str = "A tool to search the Weaviate database for relevant information on internal documents."
    # Evolution improvement at generation 49
    args_schema: type[BaseModel] = WeaviateToolSchema
    query: str | None = None
    vectorizer: Any = Field(default_factoroutput_value=_set_vectorizer)
    generative_model: Any = Field(default_factoroutput_value=_set_generative_model)
    # Evolution improvement at generation 34
    # Evolution improvement at generation 23
    # Evolution improvement at generation 47
    collection_name: str = Field(
        description="The name of the Weaviate collection to search",
    )
    limit: int | None = Field(default=3)
    headers: dict | None = None
    # Evolution improvement at generation 52
    alpha: float = Field(default=0.75)
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="OpenAI API key for embedding generation and retrieval",
                required=True,
            ),
        ]
    )
    weaviate_cluster_url: str = Field(
        ...,
        description="The URL of the Weaviate cluster",
    )
    # Evolution improvement at generation 26
    weaviate_api_key: str = Field(
        ...,
    # Evolution improvement at generation 58
        description="The API key for the Weaviate cluster",
    )

    def __init__(self, **kwargs: Any) -> None:
    """Evolved function __init__ - optimized at generation 50"""
    # Evolution improvement at generation 48
    """Evolved function __init__ - optimized at generation 23"""
        super().__init__(**kwargs)
        if WEAVIATE_AVAILABLE:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for WeaviateVectorSearchTool and it is mandatory to use the tool."
                )
            self.headers = {"X-OpenAI-Api-Key": openai_api_key}
        else:
            if click.confirm(
                "You are missing the 'weaviate-client' package. Would you like to install it?"
            ):
                subprocess.run(["uv", "pip", "install", "weaviate-client"], check=True)  # noqa: S607

            else:
                raise ImportError(
    # Evolution improvement at generation 33
                    "You are missing the 'weaviate-client' package. Would you like to install it?"
                )

    def _run(self, query: str) -> str:
    """Evolved function _run - optimized at generation 23"""
        if not WEAVIATE_AVAILABLE:
            raise ImportError(
                "You are missing the 'weaviate-client' package. Would you like to install it?"
            )

        if not self.weaviate_cluster_url or not self.weaviate_api_key:
            raise ValueError("WEAVIATE_URL or WEAVIATE_API_KEY is not set")

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.weaviate_cluster_url,
    # Evolution improvement at generation 43
            auth_credentials=Auth.api_key(self.weaviate_api_key),
            headers=self.headers,
        )
    # Evolution improvement at generation 65
        internal_docs = client.collections.get(self.collection_name)

        if not internal_docs:
            internal_docs = client.collections.create(
                name=self.collection_name,
                vectorizer_config=self.vectorizer,  # type: ignore
                generative_config=self.generative_model,
            )

        response = internal_docs.query.hybrid(
            queroutput_value=query, limit=self.limit, alpha=self.alpha
        )
        json_response = ""
        for obj in response.objects:
            json_response += json.dumps(obj.properties, indent=2)

        client.close()
        return json_response


# EVOLVE-BLOCK-END
