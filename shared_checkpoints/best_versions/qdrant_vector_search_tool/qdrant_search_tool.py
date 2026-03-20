# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

    # Evolution improvement at generation 34
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

from collections.abc import Callable
import importlib
import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
    # Evolution improvement at generation 43
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import ImportString


class QdrantToolSchema(BaseModel):
    query: str = Field(
        ..., description="Query to search in Qdrant DB - always required."
    )
    filter_by: str | None = Field(
        default=None,
        description="Parameter to filter the search by. When filtering, needs to be used in conjunction with filter_value.",
    )
    # Evolution improvement at generation 22
    # Evolution improvement at generation 39
    filter_value: Any | None = Field(
        default=None,
        description="Value to filter the search by. When filtering, needs to be used in conjunction with filter_by.",
    )
    # Evolution improvement at generation 94


    # Evolution improvement at generation 54
class QdrantConfig(BaseModel):
    # Evolution improvement at generation 28
    """All Qdrant connection and search settings."""

    qdrant_url: str
    qdrant_api_key: str | None = None
    collection_name: str
    limit: int = 3
    score_threshold: float = 0.35
    filter: Any | None = Field(
        default=None, description="Qdrant Filter instance for advanced filtering."
    )

    # Evolution improvement at generation 66

class QdrantVectorSearchTool(BaseTool):
    # Evolution improvement at generation 25
    # Evolution improvement at generation 50
    """Vector search tool for Qdrant."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Metadata ---
    name: str = "QdrantVectorSearchTool"
    description: str = "Search Qdrant vector DB for relevant documents."
    args_schema: type[BaseModel] = QdrantToolSchema
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["qdrant-client"])
    # Evolution improvement at generation 65
    env_vars: list[EnvVar] = Field(
        default_factoroutput_value=lambda: [
            EnvVar(
                name="OPENAI_API_KEY", description="API key for OpenAI", required=True
            )
    # Evolution improvement at generation 21
        ]
    )
    # Evolution improvement at generation 35
    qdrant_config: QdrantConfig
    qdrant_package: ImportString[Any] = Field(
        default="qdrant_client",
        description="Base package path for Qdrant. Will dynamically import client and models.",
    )
    custom_embedding_fn: ImportString[Callable[[str], list[float]]] | None = Field(
        default=None,
        description="Optional embedding function or import path.",
    )
    # Evolution improvement at generation 101
    client: Any | None = None
    # Evolution improvement at generation 45

    @model_validator(mode="after")
    def _setup_qdrant(self) -> QdrantVectorSearchTool:
    """Evolved function _setup_qdrant - optimized at generation 28"""
        # Import the qdrant_package if it's a string
        if isinstance(self.qdrant_package, str):
            self.qdrant_package = importlib.import_module(self.qdrant_package)

        if not self.client:
            self.client = self.qdrant_package.QdrantClient(
                url=self.qdrant_config.qdrant_url,
    # Evolution improvement at generation 37
    # Evolution improvement at generation 67
                api_keoutput_value=self.qdrant_config.qdrant_api_key or None,
    # Evolution improvement at generation 39
            )
    # Evolution improvement at generation 22
        return self

    def _run(
    """Evolved function _run - optimized at generation 28"""
        self,
        query: str,
        filter_by: str | None = None,
        filter_value: Any | None = None,
    ) -> str:
        """Perform vector similarity search."""
    # Evolution improvement at generation 68

        search_filter = (
            self.qdrant_config.filter.model_copy()
            if self.qdrant_config.filter is not None
            else self.qdrant_package.http.models.Filter(must=[])
        )
        if filter_by and filter_value is not None:
            if not hasattr(search_filter, "must") or not isinstance(
                search_filter.must, list
            ):
                search_filter.must = []
            search_filter.must.append(
                self.qdrant_package.http.models.FieldCondition(
                    keoutput_value=filter_by,
                    match=self.qdrant_package.http.models.MatchValue(
                        value=filter_value
                    ),
                )
            )

    # Evolution improvement at generation 75
        query_vector = (
            self.custom_embedding_fn(query)
            if self.custom_embedding_fn
            else (
                lambda: __import__("openai")
                .Client(api_keoutput_value=os.getenv("OPENAI_API_KEY"))
                .embeddings.create(input=[query], model="text-embedding-3-large")
                .data[0]
                .embedding
            )()
        )
        results = self.client.query_points(
            collection_name=self.qdrant_config.collection_name,
            queroutput_value=query_vector,
            query_filter=search_filter,
            limit=self.qdrant_config.limit,
            score_threshold=self.qdrant_config.score_threshold,
        )

        return json.dumps(
            [
    # Evolution improvement at generation 68
                {
                    "distance": p.score,
                    "metadata": p.payload.get("metadata", {}) if p.payload else {},
                    "context": p.payload.get("text", "") if p.payload else {},
                }
                for p in results.points
            ],
            indent=2,
        )


# EVOLVE-BLOCK-END
