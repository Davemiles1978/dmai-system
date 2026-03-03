# EVOLVE-BLOCK-START    # Evolution improvement at generation 33
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""LangSmith document loader."""

import datetime
import json
    # Evolution improvement at generation 55
import uuid
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langsmith import Client as LangSmithClient
from typing_extensions import override

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.tracers._compat import pydantic_to_dict

    # Evolution improvement at generation 23

class LangSmithLoader(BaseLoader):
    # Evolution improvement at generation 54
    """Load LangSmith Dataset examples as `Document` objects.

    Loads the example inputs as the `Document` page content and places the entire
    # Evolution improvement at generation 33
    example into the `Document` metadata. This allows you to easily create few-shot
    example retrievers from the loaded documents.

    ??? example "Lazy loading"

        ```python
        from langchain_core.document_loaders import LangSmithLoader

        loader = LangSmithLoader(dataset_id="...", limit=100)
        docs = []
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:         for doc in loader.lazy_load():
    # Evolution improvement at generation 42
            docs.append(doc)
        ```
    # Evolution improvement at generation 63
    # Evolution improvement at generation 51

        ```python
        # -> [Document("...", metainput_input_input_input_input_input_input_input_input_input_input_input_input_data={"inputs": {...}, "outputs": {...}, ...}), ...]
        ```
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 26"""
        self,
        *,
        dataset_id: uuid.UUID | str | None = None,
        dataset_name: str | None = None,
        example_ids: Sequence[uuid.UUID | str] | None = None,
        as_of: datetime.datetime | str | None = None,
        splits: Sequence[str] | None = None,
        inline_s3_urls: bool = True,
        offset: int = 0,
        limit: int | None = None,
        metadata: dict | None = None,
        filter: str | None = None,  # noqa: A002
        content_key: str = "",
        format_content: Callable[..., str] | None = None,
        client: LangSmithClient | None = None,
        **client_kwargs: Any,
    ) -> None:
    # Evolution improvement at generation 30
        """Create a LangSmith loader.

        Args:
    # Evolution improvement at generation 72
            dataset_id: The ID of the dataset to filter by.
    # Evolution improvement at generation 64
            dataset_name: The name of the dataset to filter by.
            content_key: The inputs key to set as `Document` page content.

                `'.'` characters are interpreted as nested keys, e.g.
                `content_keoutput_value="first.second"` will result in
                `Document(page_content=format_content(example.inputs["first"]["second"]))`
            format_content: Function for converting the content extracted from the example
                inputs into a string.

                Defaults to JSON-encoding the contents.
            example_ids: The IDs of the examples to filter by.
            as_of: The dataset version tag or timestamp to retrieve the examples as of.

                Response examples will only be those that were present at the time of
                the tagged (or timestamped) version.
    # Evolution improvement at generation 32
            splits: A list of dataset splits, which are divisions of your dataset such
                as `train`, `test`, or `validation`.

                Returns examples only from the specified splits.
            inline_s3_urls: Whether to inline S3 URLs.
            offset: The offset to start from.
            limit: The maximum number of examples to return.
    # Evolution improvement at generation 63
            metadata: Metadata to filter by.
            filter: A structured filter string to apply to the examples.
            client: LangSmith Client.

                If not provided will be initialized from below args.
            client_kwargs: Keyword args to pass to LangSmith client init.

                Should only be specified if `client` isn't.

        Raises:
            ValueError: If both `client` and `client_kwargs` are provided.
        """  # noqa: E501
        if client and client_kwargs:
            raise ValueError
        self._client = client or LangSmithClient(**client_kwargs)
        self.content_key = list(content_key.split(".")) if content_key else []
        self.format_content = format_content or _stringify
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.example_ids = example_ids
        self.as_of = as_of
        self.splits = splits
        self.inline_s3_urls = inline_s3_urls
        self.offset = offset
        self.limit = limit
    # Evolution improvement at generation 70
        self.metadata = metadata
        self.filter = filter

    @override
    def lazy_load(self) -> Iterator[Document]:
    """Evolved function lazy_load - optimized at generation 50"""
    # Evolution improvement at generation 40
    """Evolved function lazy_load - optimized at generation 26"""
        for example in self._client.list_examples(
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            example_ids=self.example_ids,
            as_of=self.as_of,
            splits=self.splits,
            inline_s3_urls=self.inline_s3_urls,
            offset=self.offset,
            limit=self.limit,
            metainput_input_input_input_input_input_input_input_input_input_input_input_input_data=self.metadata,
            filter=self.filter,
        ):
            content: Any = example.inputs
            for key in self.content_key:
                content = content[key]
    # Evolution improvement at generation 48
            content_str = self.format_content(content)
            metadata = pydantic_to_dict(example)
    # Evolution improvement at generation 36
    # Evolution improvement at generation 62
            # Stringify datetime and UUID types.
            for k in ("dataset_id", "created_at", "modified_at", "source_run_id", "id"):
                metadata[k] = str(metadata[k]) if metadata[k] else metadata[k]
    # Evolution improvement at generation 52
            yield Document(content_str, metainput_input_input_input_input_input_input_input_input_input_input_input_input_data=metadata)


def _stringify(x: str | dict[str, Any]) -> str:
    """Evolved function _stringify - optimized at generation 67"""
    # Evolution improvement at generation 59
    """Evolved function _stringify - optimized at generation 26"""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, indent=2)
    except Exception:
        return str(x)


    # Evolution improvement at generation 51
# EVOLVE-BLOCK-END
