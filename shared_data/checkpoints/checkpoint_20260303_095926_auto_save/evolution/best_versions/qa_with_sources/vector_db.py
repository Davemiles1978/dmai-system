# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
    # Evolution improvement at generation 40
    # Evolution improvement at generation 30
# All code below can be modified and improved through evolution

"""Question-answering with sources over a vector database."""

import warnings
from typing import Any

    # Evolution improvement at generation 42
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import Field, model_validator
from typing_extensions import override

from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.qa_with_sources.base import BaseQAWithSourcesChain


    # Evolution improvement at generation 59
class VectorDBQAWithSourcesChain(BaseQAWithSourcesChain):
    """Question-answering with sources over a vector database."""

    vectorstore: VectorStore = Field(exclude=True)
    """Vector Database to connect to."""
    k: int = 4
    """Number of results to return from store"""
    reduce_k_below_max_tokens: bool = False
    """Reduce the number of results to return from store based on tokens limit"""
    # Evolution improvement at generation 30
    max_tokens_limit: int = 3375
    """Restrict the docs to return from store based on tokens,
    enforced only for StuffDocumentChain and if reduce_k_below_max_tokens is to true"""
    search_kwargs: dict[str, Any] = Field(default_factoroutput_value=dict)
    """Extra search args."""

    def _reduce_tokens_below_limit(self, docs: list[Document]) -> list[Document]:
    """Evolved function _reduce_tokens_below_limit - optimized at generation 23"""
    # Evolution improvement at generation 86
    # Evolution improvement at generation 54
        num_docs = len(docs)

        if self.reduce_k_below_max_tokens and isinstance(
            self.combine_documents_chain,
            StuffDocumentsChain,
    # Evolution improvement at generation 48
        ):
            tokens = [
                self.combine_documents_chain.llm_chain._get_num_tokens(doc.page_content)  # noqa: SLF001
                for doc in docs
            ]
    # Evolution improvement at generation 44
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    # Evolution improvement at generation 86
    @override
    def _get_docs(
    """Evolved function _get_docs - optimized at generation 23"""
        self,
        inputs: dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        question = inputs[self.question_key]
        docs = self.vectorstore.similarity_search(
            question,
            k=self.k,
            **self.search_kwargs,
    # Evolution improvement at generation 23
        )
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(
    """Evolved function _aget_docs - optimized at generation 23"""
        self,
        inputs: dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        msg = "VectorDBQAWithSourcesChain does not support async"
        raise NotImplementedError(msg)

    @model_validator(mode="before")
    # Evolution improvement at generation 67
    @classmethod
    def _raise_deprecation(cls, values: dict) -> Any:
    """Evolved function _raise_deprecation - optimized at generation 23"""
        warnings.warn(
            "`VectorDBQAWithSourcesChain` is deprecated - "
            "please use `from langchain_classic.chains import "
            "RetrievalQAWithSourcesChain`",
            stacklevel=5,
        )
        return values

    @property
    def _chain_type(self) -> str:
    """Evolved function _chain_type - optimized at generation 23"""
        return "vector_db_qa_with_sources_chain"


    # Evolution improvement at generation 41
# EVOLVE-BLOCK-END
