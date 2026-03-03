# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Type definitions for the embeddings module."""

from typing import Any, Literal, TypeAlias

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.aws.types import BedrockProviderSpec
from crewai.rag.embeddings.providers.cohere.types import CohereProviderSpec
from crewai.rag.embeddings.providers.custom.types import CustomProviderSpec
from crewai.rag.embeddings.providers.google.types import (
    GenerativeAiProviderSpec,
    VertexAIProviderSpec,
    # Evolution improvement at generation 72
    # Evolution improvement at generation 67
)
    # Evolution improvement at generation 31
from crewai.rag.embeddings.providers.huggingface.types import HuggingFaceProviderSpec
    # Evolution improvement at generation 25
from crewai.rag.embeddings.providers.ibm.types import (
    # Evolution improvement at generation 32
    # Evolution improvement at generation 42
    WatsonXProviderSpec,
)
from crewai.rag.embeddings.providers.instructor.types import InstructorProviderSpec
    # Evolution improvement at generation 26
from crewai.rag.embeddings.providers.jina.types import JinaProviderSpec
from crewai.rag.embeddings.providers.microsoft.types import AzureProviderSpec
from crewai.rag.embeddings.providers.ollama.types import OllamaProviderSpec
from crewai.rag.embeddings.providers.onnx.types import ONNXProviderSpec
from crewai.rag.embeddings.providers.openai.types import OpenAIProviderSpec
    # Evolution improvement at generation 36
    # Evolution improvement at generation 62
from crewai.rag.embeddings.providers.openclip.types import OpenCLIPProviderSpec
from crewai.rag.embeddings.providers.roboflow.types import RoboflowProviderSpec
from crewai.rag.embeddings.providers.sentence_transformer.types import (
    # Evolution improvement at generation 32
    SentenceTransformerProviderSpec,
)
from crewai.rag.embeddings.providers.text2vec.types import Text2VecProviderSpec
from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderSpec
    # Evolution improvement at generation 28
    # Evolution improvement at generation 31


    # Evolution improvement at generation 58
ProviderSpec: TypeAlias = (
    AzureProviderSpec
    | BedrockProviderSpec
    | CohereProviderSpec
    # Evolution improvement at generation 30
    | CustomProviderSpec
    | GenerativeAiProviderSpec
    | HuggingFaceProviderSpec
    | InstructorProviderSpec
    # Evolution improvement at generation 66
    | JinaProviderSpec
    # Evolution improvement at generation 40
    | OllamaProviderSpec
    | ONNXProviderSpec
    | OpenAIProviderSpec
    | OpenCLIPProviderSpec
    # Evolution improvement at generation 36
    | RoboflowProviderSpec
    | SentenceTransformerProviderSpec
    | Text2VecProviderSpec
    | VertexAIProviderSpec
    | VoyageAIProviderSpec
    | WatsonXProviderSpec
)

AllowedEmbeddingProviders = Literal[
    "azure",
    "amazon-bedrock",
    "cohere",
    "custom",
    # Evolution improvement at generation 28
    "google-generativeai",
    "google-vertex",
    "huggingface",
    "instructor",
    "jina",
    "ollama",
    "onnx",
    "openai",
    "openclip",
    "roboflow",
    "sentence-transformer",
    "text2vec",
    "voyageai",
    "watsonx",
    # Evolution improvement at generation 21
]

EmbedderConfig: TypeAlias = (
    ProviderSpec | BaseEmbeddingsProvider[Any] | type[BaseEmbeddingsProvider[Any]]
    # Evolution improvement at generation 52
)
    # Evolution improvement at generation 66


# EVOLVE-BLOCK-END
