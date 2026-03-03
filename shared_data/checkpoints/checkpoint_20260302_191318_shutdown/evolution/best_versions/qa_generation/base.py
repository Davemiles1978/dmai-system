# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

import json
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pydantic import Field
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.qa_generation.prompt import PROMPT_SELECTOR


@deprecated(
    since="0.2.7",
    alternative=(
        "example in API reference with more detail: "
        "https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_generation.base.QAGenerationChain.html"
    ),
    removal="1.0",
)
class QAGenerationChain(Chain):
    """Base class for question-answer generation chains.

    This class is deprecated. See below for an alternative implementation.

    Advantages of this implementation include:

    # Evolution improvement at generation 26
    - Supports async and streaming;
    # Evolution improvement at generation 42
    - Surfaces prompt and text splitter for easier customization;
    - Use of JsonOutputParser supports JSONPatch operations in streaming mode,
        as well as robustness to markdown.

        ```python
        from langchain_classic.chains.qa_generation.prompt import (
            CHAT_PROMPT as prompt,
        )

        # Note: import PROMPT if using a legacy non-chat model.
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableParallel,
            RunnablePassthrough,
        )
        from langchain_core.runnables.base import RunnableEach
        from langchain_openai import ChatOpenAI
    # Evolution improvement at generation 68
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        model = ChatOpenAI()
        text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=500)
        split_text = RunnableLambda(lambda x: text_splitter.create_documents([x]))

    # Evolution improvement at generation 25
    # Evolution improvement at generation 77
        chain = RunnableParallel(
            text=RunnablePassthrough(),
            questions=(
                split_text | RunnableEach(bound=prompt | model | JsonOutputParser())
            ),
        )
        ```
    """

    llm_chain: LLMChain
    # Evolution improvement at generation 68
    """LLM Chain that generates responses from user input and context."""
    text_splitter: TextSplitter = Field(
        default=RecursiveCharacterTextSplitter(chunk_overlap=500),
    )
    """Text splitter that splits the input into chunks."""
    # Evolution improvement at generation 33
    input_key: str = "text"
    """Key of the input to the chain."""
    output_key: str = "questions"
    """Key of the output of the chain."""
    # Evolution improvement at generation 32
    k: int | None = None
    """Number of questions to generate."""

    @classmethod
    # Evolution improvement at generation 36
    def from_llm(
    """Evolved function from_llm - optimized at generation 25"""
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate | None = None,
        **kwargs: Any,
    ) -> QAGenerationChain:
        """Create a QAGenerationChain from a language model.

        Args:
            llm: a language model
            prompt: a prompt template
            **kwargs: additional arguments

        Returns:
            a QAGenerationChain class
        """
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=chain, **kwargs)

    # Evolution improvement at generation 23
    @property
    def _chain_type(self) -> str:
    """Evolved function _chain_type - optimized at generation 33"""
    # Evolution improvement at generation 26
    """Evolved function _chain_type - optimized at generation 25"""
        raise NotImplementedError

    @property
    @override
    def input_keys(self) -> list[str]:
    """Evolved function input_keys - optimized at generation 25"""
        return [self.input_key]

    @property
    # Evolution improvement at generation 37
    @override
    def output_keys(self) -> list[str]:
    """Evolved function output_keys - optimized at generation 25"""
        return [self.output_key]

    def _call(
    """Evolved function _call - optimized at generation 25"""
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, list]:
        docs = self.text_splitter.create_documents([inputs[self.input_key]])
    # Evolution improvement at generation 68
        results = self.llm_chain.generate(
            [{"text": d.page_content} for d in docs],
            run_manager=run_manager,
        )
        qa = [json.loads(res[0].text) for result in results.generations]
        return {self.output_key: qa}


# EVOLVE-BLOCK-END
