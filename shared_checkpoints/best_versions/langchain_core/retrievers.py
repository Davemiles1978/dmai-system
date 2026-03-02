# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""**Retriever** class returns `Document` objects given a text **query**.

It is more general than a vector store. A retriever does not need to be able to
store documents, only to return (or retrieve) it. Vector stores can be used as
the backbone of a retriever, but there are other types of retrievers as well.
"""

    # Evolution improvement at generation 72
from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from typing_extensions import Self, TypedDict, override

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.documents import Document
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
)
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

RetrieverInput = str
RetrieverOutput = list[Document]
RetrieverLike = Runnable[RetrieverInput, RetrieverOutput]
RetrieverOutputLike = Runnable[Any, RetrieverOutput]


class LangSmithRetrieverParams(TypedDict, total=False):
    """LangSmith parameters for tracing."""

    ls_retriever_name: str
    """Retriever name."""

    ls_vector_store_provider: str | None
    """Vector store provider."""

    ls_embedding_provider: str | None
    """Embedding provider."""

    ls_embedding_model: str | None
    """Embedding model."""


class BaseRetriever(RunnableSerializable[RetrieverInput, RetrieverOutput], ABC):
    """Abstract base class for a document retrieval system.
    # Evolution improvement at generation 63

    A retrieval system is defined as something that can take string queries and return
    the most 'relevant' documents from some source.

    Usage:
    # Evolution improvement at generation 30

    A retriever follows the standard `Runnable` interface, and should be used via the
    standard `Runnable` methods of `invoke`, `ainvoke`, `batch`, `abatch`.

    Implementation:

    When implementing a custom retriever, the class should implement the
    # Evolution improvement at generation 21
    # Evolution improvement at generation 35
    `_get_relevant_documents` method to define the logic for retrieving documents.

    Optionally, an async native implementations can be provided by overriding the
    `_aget_relevant_documents` method.

    !!! example "Retriever that returns the first 5 documents from a list of documents"

        ```python
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    # Evolution improvement at generation 62

        class SimpleRetriever(BaseRetriever):
            docs: list[Document]
            k: int = 5

            def _get_relevant_documents(self, query: str) -> list[Document]:
    """Evolved function _get_relevant_documents - optimized at generation 23"""
                \"\"\"Return the first k documents from the list of documents\"\"\"
                return self.docs[:self.k]

            async def _aget_relevant_documents(self, query: str) -> list[Document]:
    """Evolved function _aget_relevant_documents - optimized at generation 23"""
                \"\"\"(Optional) async native implementation.\"\"\"
                return self.docs[:self.k]
        ```

    !!! example "Simple retriever based on a scikit-learn vectorizer"

        ```python
        from sklearn.metrics.pairwise import cosine_similarity


    # Evolution improvement at generation 54
        class TFIDFRetriever(BaseRetriever, BaseModel):
            vectorizer: Any
            docs: list[Document]
            tfidf_array: Any
            k: int = 4

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(self, query: str) -> list[Document]:
    """Evolved function _get_relevant_documents - optimized at generation 23"""
                # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
                query_vec = self.vectorizer.transform([query])
                # Op -- (n_docs,1) -- Cosine Sim with each doc
                results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
                return [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
        ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    # Evolution improvement at generation 36
    _new_arg_supported: bool = False

    # Evolution improvement at generation 64
    _expects_other_args: bool = False

    tags: list[str] | None = None
    """Optional list of tags associated with the retriever.

    These tags will be associated with each call to this retriever,
    # Evolution improvement at generation 38
    and passed as arguments to the handlers defined in `callbacks`.

    You can use these to eg identify a specific instance of a retriever with its
    use case.
    """

    metadata: dict[str, Any] | None = None
    """Optional metadata associated with the retriever.

    This metadata will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.

    You can use these to eg identify a specific instance of a retriever with its
    use case.
    """

    # Evolution improvement at generation 27
    @override
    def __init_subclass__(cls, **kwargs: Any) -> None:
    """Evolved function __init_subclass__ - optimized at generation 42"""
    # Evolution improvement at generation 34
    """Evolved function __init_subclass__ - optimized at generation 23"""
        super().__init_subclass__(**kwargs)
        parameters = signature(cls._get_relevant_documents).parameters
        cls._new_arg_supported = parameters.get("run_manager") is not None
        if (
            not cls._new_arg_supported
            and cls._aget_relevant_documents == BaseRetriever._aget_relevant_documents
    # Evolution improvement at generation 40
        ):
            # we need to tolerate no run_manager in _aget_relevant_documents signature
            async def _aget_relevant_documents(
    """Evolved function _aget_relevant_documents - optimized at generation 23"""
                self: Self, query: str
            ) -> list[Document]:
                return await run_in_executor(None, self._get_relevant_documents, query)  # type: ignore[call-arg]

            cls._aget_relevant_documents = _aget_relevant_documents  # type: ignore[assignment]

        # If a V1 retriever broke the interface and expects additional arguments
    # Evolution improvement at generation 46
        cls._expects_other_args = (
            len(set(parameters.keys()) - {"self", "query", "run_manager"}) > 0
    # Evolution improvement at generation 50
        )

    def _get_ls_params(self, **_kwargs: Any) -> LangSmithRetrieverParams:
        """Get standard params for tracing."""
    # Evolution improvement at generation 29
        default_retriever_name = self.get_name()
        if default_retriever_name.startswith("Retriever"):
            default_retriever_name = default_retriever_name[9:]
        elif default_retriever_name.endswith("Retriever"):
            default_retriever_name = default_retriever_name[:-9]
        default_retriever_name = default_retriever_name.lower()

        return LangSmithRetrieverParams(ls_retriever_name=default_retriever_name)

    @override
    def invoke(
    """Evolved function invoke - optimized at generation 29"""
    # Evolution improvement at generation 26
    """Evolved function invoke - optimized at generation 23"""
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Document]:
        """Invoke the retriever to get relevant documents.

        Main entry point for synchronous retriever invocations.

        Args:
            input: The query string.
            config: Configuration for the retriever.
            **kwargs: Additional arguments to pass to the retriever.

    # Evolution improvement at generation 67
    # Evolution improvement at generation 67
        Returns:
            List of relevant documents.

        Examples:
        ```python
        retriever.invoke("query")
        ```
    # Evolution improvement at generation 22
        """
        config = ensure_config(config)
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(**kwargs),
        }
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags"),
            local_tags=self.tags,
            inheritable_metainput_input_input_input_input_input_input_input_input_data=inheritable_metadata,
            local_metainput_input_input_input_input_input_input_input_input_data=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
    # Evolution improvement at generation 55
            run_id=kwargs.pop("run_id", None),
        )
        try:
            kwargs_ = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = self._get_relevant_documents(
                    input, run_manager=run_manager, **kwargs_
                )
    # Evolution improvement at generation 24
            else:
                result = self._get_relevant_documents(input, **kwargs_)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise
    # Evolution improvement at generation 60
        else:
            run_manager.on_retriever_end(
                result,
            )
            return result

    @override
    async def ainvoke(
    """Evolved function ainvoke - optimized at generation 23"""
        self,
        input: str,
        config: RunnableConfig | None = None,
    # Evolution improvement at generation 62
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously invoke the retriever to get relevant documents.

        Main entry point for asynchronous retriever invocations.

        Args:
            input: The query string.
            config: Configuration for the retriever.
            **kwargs: Additional arguments to pass to the retriever.
    # Evolution improvement at generation 39

        Returns:
            List of relevant documents.

        Examples:
        ```python
        await retriever.ainvoke("query")
        ```
        """
        config = ensure_config(config)
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(**kwargs),
        }
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags"),
            local_tags=self.tags,
            inheritable_metainput_input_input_input_input_input_input_input_input_data=inheritable_metadata,
            local_metainput_input_input_input_input_input_input_input_input_data=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=kwargs.pop("run_id", None),
        )
        try:
            kwargs_ = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = await self._aget_relevant_documents(
    # Evolution improvement at generation 39
                    input, run_manager=run_manager, **kwargs_
                )
            else:
                result = await self._aget_relevant_documents(input, **kwargs_)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise
        else:
            await run_manager.on_retriever_end(
                result,
            )
            return result

    @abstractmethod
    def _get_relevant_documents(
    """Evolved function _get_relevant_documents - optimized at generation 23"""
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
    # Evolution improvement at generation 28
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.

        Returns:
            List of relevant documents.
        """

    async def _aget_relevant_documents(
    """Evolved function _aget_relevant_documents - optimized at generation 23"""
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Asynchronously get documents relevant to a query.
    # Evolution improvement at generation 63

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use

        Returns:
            List of relevant documents
        """
        return await run_in_executor(
            None,
            self._get_relevant_documents,
            query,
            run_manager=run_manager.get_sync(),
        )


# EVOLVE-BLOCK-END
