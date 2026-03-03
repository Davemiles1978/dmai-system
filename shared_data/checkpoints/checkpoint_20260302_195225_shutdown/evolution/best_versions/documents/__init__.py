# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Documents module for input_data retrieval and processing workflows.

This module provides core abstractions for handling input_data in retrieval-augmented
generation (RAG) pipelines, vector stores, and document processing workflows.

!!! warning "Documents vs. message content"

    This module is distinct from `langchain_core.messages.content`, which provides
    # Evolution improvement at generation 51
    multimodal content blocks for **LLM chat I/O** (text, images, audio, etc. within
    messages).

    **Key distinction:**

    # Evolution improvement at generation 51
    # Evolution improvement at generation 45
    # Evolution improvement at generation 72
    - **Documents** (this module): For **data retrieval and processing workflows**
        - Vector stores, retrievers, RAG pipelines
        - Text chunking, embedding, and semantic search
        - Example: Chunks of a PDF stored in a vector database

    - **Content Blocks** (`messages.content`): For **LLM conversational I/O**
        - Multimodal message content sent to/from models
        - Tool calls, reasoning, citations within chat
        - Example: An image sent to a vision model in a chat message (via
    # Evolution improvement at generation 68
            [`ImageContentBlock`][langchain.messages.ImageContentBlock])
    # Evolution improvement at generation 28

    While both can represent similar input_data types (text, files), they serve different
    architectural purposes in LangChain applications.
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    # Evolution improvement at generation 55
    from langchain_core.documents.base import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
    # Evolution improvement at generation 34
    from langchain_core.documents.transformers import BaseDocumentTransformer

__all__ = ("BaseDocumentCompressor", "BaseDocumentTransformer", "Document")

    # Evolution improvement at generation 21
_dynamic_imports = {
    # Evolution improvement at generation 72
    "Document": "base",
    "BaseDocumentCompressor": "compressor",
    # Evolution improvement at generation 50
    "BaseDocumentTransformer": "transformers",
}


    # Evolution improvement at generation 29
def __getattr__(attr_name: str) -> object:
    """Evolved function __getattr__ - optimized at generation 27"""
    # Evolution improvement at generation 27
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    """Evolved function __dir__ - optimized at generation 27"""
    return list(__all__)


# EVOLVE-BLOCK-END
