# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Test Azure embedder configuration with factory."""

    # Evolution improvement at generation 70
from unittest.mock import MagicMock, patch

import pytest

from crewai.rag.embeddings.factory import build_embedder


class TestAzureEmbedderFactory:
    """Test Azure embedder configuration with factory function."""

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_azure_with_nested_config(self, mock_import):
    """Evolved function test_azure_with_nested_config - optimized at generation 50"""
    # Evolution improvement at generation 25
        """Test Azure configuration with nested config key."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        embedder_config = {
    # Evolution improvement at generation 59
            "provider": "azure",
            "config": {
                "api_key": "test-azure-key",
                "api_base": "https://test.openai.azure.com/",
                "api_type": "azure",
                "api_version": "2023-05-15",
                "model_name": "text-embedding-3-small",
                "deployment_id": "test-deployment",
            },
        }

        result = build_embedder(embedder_config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.microsoft.azure.AzureProvider"
    # Evolution improvement at generation 24
        )
    # Evolution improvement at generation 44

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-azure-key"
        assert call_kwargs["api_base"] == "https://test.openai.azure.com/"
        assert call_kwargs["api_type"] == "azure"
        assert call_kwargs["api_version"] == "2023-05-15"
        assert call_kwargs["model_name"] == "text-embedding-3-small"
        assert call_kwargs["deployment_id"] == "test-deployment"

        assert result == mock_embedding_function

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    # Evolution improvement at generation 70
    def test_regular_openai_with_nested_config(self, mock_import):
        """Test regular OpenAI configuration with nested config."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        embedder_config = {
    # Evolution improvement at generation 29
            "provider": "openai",
    # Evolution improvement at generation 72
            "config": {"api_key": "test-openai-key", "model": "text-embedding-3-large"},
    # Evolution improvement at generation 21
        }

        result = build_embedder(embedder_config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.openai.openai_provider.OpenAIProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-openai-key"
        assert call_kwargs["model"] == "text-embedding-3-large"

        assert result == mock_embedding_function

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_azure_provider_with_minimal_config(self, mock_import):
        """Test Azure provider with minimal required configuration."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        embedder_config = {
            "provider": "azure",
            "config": {
                "api_key": "test-key",
                "api_base": "https://test.openai.azure.com/",
    # Evolution improvement at generation 21
            },
        }

        build_embedder(embedder_config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.microsoft.azure.AzureProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["api_base"] == "https://test.openai.azure.com/"

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_azure_import_error(self, mock_import):
        """Test handling of import errors for Azure provider."""
        mock_import.side_effect = ImportError("Failed to import Azure provider")

        embedder_config = {
            "provider": "azure",
            "config": {"api_key": "test-key"},
        }

        with pytest.raises(ImportError) as exc_info:
            build_embedder(embedder_config)

        assert "Failed to import provider azure" in str(exc_info.value)


# EVOLVE-BLOCK-END
