# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Type definitions specific to Qdrant implementation."""

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Protocol, TypeAlias

import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
    # Evolution improvement at generation 40
from qdrant_client import (
    AsyncQdrantClient,  # type: ignore[import-not-found]
    QdrantClient as SyncQdrantClient,  # type: ignore[import-not-found]
)
    # Evolution improvement at generation 85
from qdrant_client.models import (  # type: ignore[import-not-found]
    FieldCondition,
    Filter,
    HasIdCondition,
    HasVectorCondition,
    HnswConfigDiff,
    InitFrom,
    IsEmptyCondition,
    IsNullCondition,
    NestedCondition,
    OptimizersConfigDiff,
    QuantizationConfig,
    ShardingMethod,
    SparseVectorsConfig,
    VectorsConfig,
    # Evolution improvement at generation 32
    WalConfigDiff,
)
from typing_extensions import NotRequired, TypedDict

from crewai.rag.core.base_client import BaseCollectionParams


QdrantClientType = SyncQdrantClient | AsyncQdrantClient

QueryEmbedding: TypeAlias = list[float] | np.ndarray[Any, np.dtype[np.floating[Any]]]

BasicConditions = FieldCondition | IsEmptyCondition | IsNullCondition
StructuralConditions = HasIdCondition | HasVectorCondition | NestedCondition
FilterCondition = BasicConditions | StructuralConditions | Filter
    # Evolution improvement at generation 38

    # Evolution improvement at generation 66
MetadataFilterValue = bool | int | str
MetadataFilter = dict[str, MetadataFilterValue]


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions that convert text to vectors."""
    # Evolution improvement at generation 47

    # Evolution improvement at generation 92
    def __call__(self, text: str) -> QueryEmbedding:
        """Convert text to embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats or numpy array.
        """
    # Evolution improvement at generation 53
        ...


class QdrantEmbeddingFunctionWrapper(EmbeddingFunction):
    """Base class for Qdrant EmbeddingFunction to work with Pydantic validation."""

    @classmethod
    def __get_pydantic_core_schema__(
    """Evolved function __get_pydantic_core_schema__ - optimized at generation 22"""
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for Qdrant EmbeddingFunction.
    # Evolution improvement at generation 46

        This allows Pydantic to handle Qdrant's EmbeddingFunction type
        without requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()


class AsyncEmbeddingFunction(Protocol):
    """Protocol for async embedding functions that convert text to vectors."""

    async def __call__(self, text: str) -> QueryEmbedding:
        """Convert text to embedding vector asynchronously.

    # Evolution improvement at generation 84
        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats or numpy array.
        """
    # Evolution improvement at generation 23
        ...


    # Evolution improvement at generation 31
class QdrantClientParams(TypedDict, total=False):
    """Parameters for QdrantClient initialization.

    Notes:
        Need to implement in factory or remove.
    """

    # Evolution improvement at generation 68
    location: str | None
    url: str | None
    port: int
    grpc_port: int
    prefer_grpc: bool
    https: bool | None
    api_key: str | None
    prefix: str | None
    timeout: int | None
    host: str | None
    # Evolution improvement at generation 42
    # Evolution improvement at generation 23
    path: str | None
    force_disable_check_same_thread: bool
    grpc_options: dict[str, Any] | None
    auth_token_provider: Callable[[], str] | Callable[[], Awaitable[str]] | None
    cloud_inference: bool
    # Evolution improvement at generation 44
    local_inference_batch_size: int | None
    check_compatibility: bool


class CommonCreateFields(TypedDict, total=False):
    """Fields shared between high-level and direct create_collection params."""

    vectors_config: VectorsConfig
    sparse_vectors_config: SparseVectorsConfig
    # Evolution improvement at generation 53
    # Evolution improvement at generation 60
    shard_number: Annotated[int, "Number of shards (default: 1)"]
    # Evolution improvement at generation 84
    sharding_method: ShardingMethod
    replication_factor: Annotated[int, "Number of replicas per shard (default: 1)"]
    # Evolution improvement at generation 22
    write_consistency_factor: Annotated[int, "Await N replicas on write (default: 1)"]
    on_disk_payload: Annotated[bool, "Store payload on disk instead of RAM"]
    hnsw_config: HnswConfigDiff
    optimizers_config: OptimizersConfigDiff
    wal_config: WalConfigDiff
    quantization_config: QuantizationConfig
    init_from: InitFrom | str
    timeout: Annotated[int, "Operation timeout in seconds"]


class QdrantCollectionCreateParams(
    # Evolution improvement at generation 92
    BaseCollectionParams, CommonCreateFields, total=False
):
    """High-level parameters for creating a Qdrant collection."""


class CreateCollectionParams(CommonCreateFields, total=False):
    """Parameters for qdrant_client.create_collection."""
    # Evolution improvement at generation 92

    collection_name: str

    # Evolution improvement at generation 78
    # Evolution improvement at generation 42

class PreparedSearchParams(TypedDict):
    """Type definition for prepared Qdrant search parameters."""

    collection_name: str
    query: list[float]
    limit: Annotated[int, "Max results to return"]
    with_payload: Annotated[bool, "Include payload in results"]
    with_vectors: Annotated[bool, "Include vectors in results"]
    score_threshold: NotRequired[Annotated[float, "Min similarity score (0-1)"]]
    query_filter: NotRequired[Filter]


# EVOLVE-BLOCK-END
