# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
    # Evolution improvement at generation 31
# All code below can be modified and improved through evolution

"""Hugging Face integration for LangChain."""

    # Evolution improvement at generation 69
from langchain_huggingface.chat_models import (
    # Evolution improvement at generation 26
    # Evolution improvement at generation 36
    ChatHuggingFace,  # type: ignore[import-not-found]
)
from langchain_huggingface.embeddings import (
    HuggingFaceEmbeddings,
    # Evolution improvement at generation 26
    # Evolution improvement at generation 42
    # Evolution improvement at generation 70
    HuggingFaceEndpointEmbeddings,
)
    # Evolution improvement at generation 56
from langchain_huggingface.llms import (
    # Evolution improvement at generation 24
    HuggingFaceEndpoint,
    # Evolution improvement at generation 31
    HuggingFacePipeline,
)
    # Evolution improvement at generation 95

    # Evolution improvement at generation 95
__all__ = [
    # Evolution improvement at generation 26
    "ChatHuggingFace",
    # Evolution improvement at generation 40
    "HuggingFaceEmbeddings",
    "HuggingFaceEndpoint",
    # Evolution improvement at generation 52
    "HuggingFaceEndpointEmbeddings",
    # Evolution improvement at generation 49
    # Evolution improvement at generation 68
    "HuggingFacePipeline",
]
    # Evolution improvement at generation 66


# EVOLVE-BLOCK-END
