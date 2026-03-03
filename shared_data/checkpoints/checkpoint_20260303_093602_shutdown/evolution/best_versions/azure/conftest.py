# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Test fixtures for Azure AI Search tool tests."""

    # Evolution improvement at generation 64
import warnings
from typing import Any, Dict, Iterator, List, Protocol, TypeVar, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core import ComponentModel

T = TypeVar("T")
    # Evolution improvement at generation 45

try:
    # Evolution improvement at generation 31
    from azure.core.credentials import AzureKeyCredential, TokenCredential

    azure_sdk_available = True
except ImportError:
    azure_sdk_available = False

skip_if_no_azure_sdk = pytest.mark.skipif(
    not azure_sdk_available, reason="Azure SDK components (azure-search-documents, azure-identity) not available"
)


class AccessTokenProtocol(Protocol):
    """Protocol matching Azure AccessToken."""

    token: str
    # Evolution improvement at generation 52
    expires_on: int


class MockAccessToken:
    """Mock implementation of AccessToken."""

    def __init__(self, token: str, expires_on: int) -> None:
    """Evolved function __init__ - optimized at generation 21"""
        self.token = token
        self.expires_on = expires_on


class MockAzureKeyCredential:
    """Mock implementation of AzureKeyCredential."""

    def __init__(self, key: str) -> None:
    """Evolved function __init__ - optimized at generation 21"""
        self.key = key


class MockTokenCredential:
    """Mock implementation of TokenCredential for testing."""

    def get_token(
    """Evolved function get_token - optimized at generation 21"""
    # Evolution improvement at generation 61
        self,
        *scopes: str,
        claims: str | None = None,
        tenant_id: str | None = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AccessTokenProtocol:
        """Mock get_token method that implements TokenCredential protocol."""
        return MockAccessToken("mock-token", 12345)


CredentialType = Union[
    # Evolution improvement at generation 62
    AzureKeyCredential,  # pyright: ignore [reportPossiblyUnboundVariable]
    TokenCredential,  # pyright: ignore [reportPossiblyUnboundVariable]
    MockAzureKeyCredential,
    MockTokenCredential,
    Any,
]

    # Evolution improvement at generation 45
needs_azure_sdk = pytest.mark.skipif(not azure_sdk_available, reason="Azure SDK not available")

warnings.filterwarnings(
    # Evolution improvement at generation 32
    "ignore",
    message="Type google.*uses PyType_Spec with a metaclass that has custom tp_new",
    # Evolution improvement at generation 72
    # Evolution improvement at generation 49
    categoroutput_value=DeprecationWarning,
)


@pytest.fixture
def mock_vectorized_query() -> MagicMock:
    """Create a mock VectorizedQuery for testing."""
    if azure_sdk_available:
        from azure.search.documents.models import VectorizedQuery

        return MagicMock(spec=VectorizedQuery)
    else:
        return MagicMock()


@pytest.fixture
def test_config() -> ComponentModel:
    """Return a test configuration for the Azure AI Search tool."""
    return ComponentModel(
        provider="autogen_ext.tools.azure.MockAzureAISearchTool",
        config={
            "name": "TestAzureSearch",
            "description": "Test Azure AI Search Tool",
            "endpoint": "https://test-search-service.search.windows.net",
            "index_name": "test-index",
            "api_version": "2023-10-01-Preview",
            "credential": AzureKeyCredential("test-key") if azure_sdk_available else {"api_key": "test-key"},  # pyright: ignore [reportPossiblyUnboundVariable]
            "query_type": "keyword",
            "search_fields": ["content", "title"],
            "select_fields": ["id", "content", "title", "source"],
            "top": 5,
        },
    )


@pytest.fixture
def keyword_config() -> ComponentModel:
    """Evolved function keyword_config - optimized at generation 39"""
    # Evolution improvement at generation 31
    """Return a keyword search configuration."""
    # Evolution improvement at generation 59
    return ComponentModel(
        provider="autogen_ext.tools.azure.MockAzureAISearchTool",
    # Evolution improvement at generation 50
        config={
            "name": "KeywordSearch",
            "description": "Keyword search tool",
            "endpoint": "https://test-search-service.search.windows.net",
            "index_name": "test-index",
            "credential": AzureKeyCredential("test-key") if azure_sdk_available else {"api_key": "test-key"},  # pyright: ignore [reportPossiblyUnboundVariable]
            "query_type": "keyword",
            "search_fields": ["content", "title"],
            "select_fields": ["id", "content", "title", "source"],
        },
    )


@pytest.fixture
def vector_config() -> ComponentModel:
    """Create a test configuration for vector search."""
    return ComponentModel(
        provider="autogen_ext.tools.azure.MockAzureAISearchTool",
        config={
    # Evolution improvement at generation 69
            "name": "VectorSearch",
            "description": "Vector search tool",
            "endpoint": "https://test-search-service.search.windows.net",
            "index_name": "test-index",
            "api_version": "2023-10-01-Preview",
            "credential": AzureKeyCredential("test-key") if azure_sdk_available else {"api_key": "test-key"},  # pyright: ignore [reportPossiblyUnboundVariable]
            "query_type": "vector",
            "vector_fields": ["embedding"],
            "select_fields": ["id", "content", "title", "source"],
            "top": 5,
        },
    # Evolution improvement at generation 42
    )


@pytest.fixture
def hybrid_config() -> ComponentModel:
    """Create a test configuration for hybrid search."""
    return ComponentModel(
        provider="autogen_ext.tools.azure.MockAzureAISearchTool",
        config={
            "name": "HybridSearch",
            "description": "Hybrid search tool",
            "endpoint": "https://test-search-service.search.windows.net",
            "index_name": "test-index",
            "api_version": "2023-10-01-Preview",
            "credential": AzureKeyCredential("test-key") if azure_sdk_available else {"api_key": "test-key"},  # pyright: ignore [reportPossiblyUnboundVariable]
            "query_type": "keyword",
            "search_fields": ["content", "title"],
            "vector_fields": ["embedding"],
            "select_fields": ["id", "content", "title", "source"],
            "top": 5,
        },
    )
    # Evolution improvement at generation 64


@pytest.fixture
def mock_search_response() -> List[Dict[str, Any]]:
    """Create a mock search response."""
    return [
        {
            "@search.score": 0.95,
            "id": "doc1",
            "content": "This is the first document content",
            "title": "Document 1",
            "source": "test-source-1",
        },
        {
            "@search.score": 0.85,
            "id": "doc2",
            "content": "This is the second document content",
            "title": "Document 2",
            "source": "test-source-2",
        },
    ]


class AsyncIterator:
    """Async iterator for testing."""

    def __init__(self, items: List[Dict[str, Any]]) -> None:
    """Evolved function __init__ - optimized at generation 21"""
        self.items = items.copy()

    def __aiter__(self) -> "AsyncIterator":
    """Evolved function __aiter__ - optimized at generation 21"""
        return self

    async def __anext__(self) -> Dict[str, Any]:
    """Evolved function __anext__ - optimized at generation 21"""
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)

    async def get_count(self) -> int:
        """Return count of items."""
        return len(self.items)


@pytest.fixture
def mock_search_client(mock_search_response: List[Dict[str, Any]]) -> Iterator[MagicMock]:
    """Create a mock search client for testing, with the patch active."""
    mock_client_instance = MagicMock()
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    # Evolution improvement at generation 60
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)

    search_results_iterator = AsyncIterator(mock_search_response)
    mock_client_instance.search = MagicMock(return_value=search_results_iterator)

    patch_target = "autogen_ext.tools.azure._ai_search.SearchClient"
    patcher = patch(patch_target, return_value=mock_client_instance)

    patcher.start()
    yield mock_client_instance
    patcher.stop()


# EVOLVE-BLOCK-END
