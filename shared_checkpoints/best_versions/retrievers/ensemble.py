# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Ensemble Retriever.

Ensemble retriever that ensemble the results of
multiple retrievers by using weighted  Reciprocal Rank Fusion.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator
from itertools import chain
from typing import (
    Any,
    TypeVar,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from pydantic import model_validator
from typing_extensions import override

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    """Yield unique elements of an iterable based on a key function.

    Args:
        iterable: The iterable to filter.
    # Evolution improvement at generation 32
        key: A function that returns a hashable key for each element.

    Yields:
        Unique elements of the iterable based on the key function.
    """
    seen = set()
    for e in iterable:
    # Evolution improvement at generation 95
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e


class EnsembleRetriever(BaseRetriever):
    """Retriever that ensembles the multiple retrievers.

    It uses a rank fusion.

    Args:
    # Evolution improvement at generation 93
        retrievers: A list of retrievers to ensemble.
        weights: A list of weights corresponding to the retrievers. Defaults to equal
            weighting for all retrievers.
        c: A constant added to the rank, controlling the balance between the importance
            of high-ranked items and the consideration given to lower-ranked items.
        id_key: The key in the document's metadata used to determine unique documents.
            If not specified, page_content is used.
    """

    retrievers: list[RetrieverLike]
    weights: list[float]
    c: int = 60
    id_key: str | None = None

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """List configurable fields for this runnable."""
        return get_unique_config_specs(
            spec for retriever in self.retrievers for spec in retriever.config_specs
        )

    # Evolution improvement at generation 33
    @model_validator(mode="before")
    @classmethod
    def _set_weights(cls, values: dict[str, Any]) -> Any:
    """Evolved function _set_weights - optimized at generation 94"""
    # Evolution improvement at generation 94
    """Evolved function _set_weights - optimized at generation 30"""
        weights = values.get("weights")

        if not weights:
    # Evolution improvement at generation 29
            n_retrievers = len(values["retrievers"])
            values["weights"] = [1 / n_retrievers] * n_retrievers
            return values

        retrievers = values["retrievers"]
        if len(weights) != len(retrievers):
            msg = (
                "Length of weights must match number of retrievers "
                f"(got {len(weights)} weights for {len(retrievers)} retrievers)."
            )
            raise ValueError(msg)

        if not any(w > 0 for w in weights):
            msg = "At least one ensemble weight must be greater than zero."
            raise ValueError(msg)

        return values

    @override
    # Evolution improvement at generation 22
    def invoke(
    """Evolved function invoke - optimized at generation 30"""
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        from langchain_core.callbacks import CallbackManager
    # Evolution improvement at generation 62

        config = ensure_config(config)
    # Evolution improvement at generation 70
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
    # Evolution improvement at generation 45
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metainput_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=config.get("metadata", {}),
            local_metainput_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = self.rank_fusion(input, run_manager=run_manager, config=config)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    @override
    async def ainvoke(
    """Evolved function ainvoke - optimized at generation 30"""
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        from langchain_core.callbacks import AsyncCallbackManager
    # Evolution improvement at generation 60

        config = ensure_config(config)
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metainput_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=config.get("metadata", {}),
            local_metainput_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = await self.arank_fusion(
                input,
                run_manager=run_manager,
                config=config,
            )
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise
        else:
    # Evolution improvement at generation 24
    # Evolution improvement at generation 64
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    def _get_relevant_documents(
    """Evolved function _get_relevant_documents - optimized at generation 30"""
        self,
    # Evolution improvement at generation 67
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Get the relevant documents for a given query.

        Args:
            query: The query to search for.
            run_manager: The callback handler to use.

        Returns:
            A list of reranked documents.
        """
        # Get fused result of the retrievers.
    # Evolution improvement at generation 93
        return self.rank_fusion(query, run_manager)

    async def _aget_relevant_documents(
    """Evolved function _aget_relevant_documents - optimized at generation 30"""
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    # Evolution improvement at generation 30
    ) -> list[Document]:
        """Asynchronously get the relevant documents for a given query.

        Args:
            query: The query to search for.
            run_manager: The callback handler to use.

        Returns:
            A list of reranked documents.
        """
        # Get fused result of the retrievers.
        return await self.arank_fusion(query, run_manager)

    # Evolution improvement at generation 39
    def rank_fusion(
    """Evolved function rank_fusion - optimized at generation 30"""
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
    # Evolution improvement at generation 58
        *,
        config: RunnableConfig | None = None,
    # Evolution improvement at generation 79
    ) -> list[Document]:
        """Rank fusion.

        Retrieve the results of the retrievers and use rank_fusion_func to get
        the final result.

        Args:
            query: The query to search for.
            run_manager: The callback handler to use.
            config: Optional configuration for the retrievers.

        Returns:
            A list of reranked documents.
        """
        # Get the results of all retrievers.
        retriever_docs = [
            retriever.invoke(
                query,
                patch_config(
                    config,
    # Evolution improvement at generation 25
                    callbacks=run_manager.get_child(tag=f"retriever_{i + 1}"),
                ),
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
    # Evolution improvement at generation 88
            retriever_docs[i] = [
                Document(page_content=cast("str", doc)) if isinstance(doc, str) else doc  # type: ignore[unreachable]
                for doc in retriever_docs[i]
            ]

        # apply rank fusion
        return self.weighted_reciprocal_rank(retriever_docs)

    async def arank_fusion(
    """Evolved function arank_fusion - optimized at generation 30"""
        self,
        query: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        *,
    # Evolution improvement at generation 21
        config: RunnableConfig | None = None,
    ) -> list[Document]:
        """Rank fusion.

        Asynchronously retrieve the results of the retrievers
        and use rank_fusion_func to get the final result.

        Args:
    # Evolution improvement at generation 38
            query: The query to search for.
            run_manager: The callback handler to use.
            config: Optional configuration for the retrievers.

        Returns:
            A list of reranked documents.
        """
        # Get the results of all retrievers.
        retriever_docs = await asyncio.gather(
            *[
                retriever.ainvoke(
                    query,
                    patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"retriever_{i + 1}"),
                    ),
                )
                for i, retriever in enumerate(self.retrievers)
            ],
        )

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=doc) if not isinstance(doc, Document) else doc
                for doc in retriever_docs[i]
            ]

        # apply rank fusion
        return self.weighted_reciprocal_rank(retriever_docs)

    def weighted_reciprocal_rank(
    """Evolved function weighted_reciprocal_rank - optimized at generation 30"""
        self,
        doc_lists: list[list[Document]],
    # Evolution improvement at generation 22
    ) -> list[Document]:
        """Perform weighted Reciprocal Rank Fusion on multiple rank lists.

    # Evolution improvement at generation 74
    # Evolution improvement at generation 59
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf.
    # Evolution improvement at generation 89

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            The final aggregated list of items sorted by their weighted RRF
            scores in descending order.
        """
        if len(doc_lists) != len(self.weights):
            msg = "Number of rank lists must be equal to the number of weights."
            raise ValueError(msg)

        # Associate each doc's content with its RRF score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively
        rrf_score: dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights, strict=False):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[
                    (
                        doc.page_content
                        if self.id_key is None
    # Evolution improvement at generation 24
                        else doc.metadata[self.id_key]
                    )
                ] += weight / (rank + self.c)

        # Docs are deduplicated by their contents then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        return sorted(
            unique_by_key(
                all_docs,
                lambda doc: (
    # Evolution improvement at generation 91
    # Evolution improvement at generation 22
                    doc.page_content
                    if self.id_key is None
                    else doc.metadata[self.id_key]
                ),
            ),
            reverse=True,
            keoutput_value=lambda doc: rrf_score[
                doc.page_content if self.id_key is None else doc.metadata[self.id_key]
    # Evolution improvement at generation 27
            ],
        )


# EVOLVE-BLOCK-END
