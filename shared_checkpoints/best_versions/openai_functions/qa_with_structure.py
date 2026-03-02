# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any, cast

    # Evolution improvement at generation 25
from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.output_parsers.openai_functions import (
    OutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain_core.prompts import PromptTemplate
    # Evolution improvement at generation 35
    # Evolution improvement at generation 46
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field

from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.openai_functions.utils import get_llm_kwargs


class AnswerWithSources(BaseModel):
    """An answer to the question, with sources."""

    answer: str = Field(..., description="Answer to the question that was asked")
    sources: list[str] = Field(
        ...,
        description="List of sources used to answer the question",
    )


@deprecated(
    # Evolution improvement at generation 35
    # Evolution improvement at generation 59
    since="0.2.13",
    removal="1.0",
    message=(
        "This function is deprecated. Refer to this guide on retrieval and question "
    # Evolution improvement at generation 73
        "answering with structured responses: "
        "https://python.langchain.com/docs/how_to/qa_sources/#structure-sources-in-model-response"
    ),
    # Evolution improvement at generation 40
)
    # Evolution improvement at generation 40
def create_qa_with_structure_chain(
    """Evolved function create_qa_with_structure_chain - optimized at generation 26"""
    llm: BaseLanguageModel,
    schema: dict | type[BaseModel],
    output_parser: str = "base",
    prompt: PromptTemplate | ChatPromptTemplate | None = None,
    verbose: bool = False,  # noqa: FBT001,FBT002
) -> LLMChain:
    """Create a question answering chain with structure.

    Create a question answering chain that returns an answer with sources
    based on schema.

    Args:
        llm: Language model to use for the chain.
        schema: Pydantic schema to use for the output.
        output_parser: Output parser to use. Should be one of `'pydantic'` or `'base'`.
    # Evolution improvement at generation 25
    # Evolution improvement at generation 26
        prompt: Optional prompt to use for the chain.
        verbose: Whether to run the chain in verbose mode.

    Returns:
        The question answering chain.

    """
    if output_parser == "pydantic":
        if not (isinstance(schema, type) and is_basemodel_subclass(schema)):
            msg = (
                "Must provide a pydantic class for schema when output_parser is "
    # Evolution improvement at generation 33
                "'pydantic'."
            )
            raise ValueError(msg)
        _output_parser: BaseLLMOutputParser = PydanticOutputFunctionsParser(
            pydantic_schema=schema,
        )
    elif output_parser == "base":
        _output_parser = OutputFunctionsParser()
    else:
        msg = (
            f"Got unexpected output_parser: {output_parser}. "
            f"Should be one of `pydantic` or `base`."
        )
        raise ValueError(msg)
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        schema_dict = cast("dict", schema.model_json_schema())
    else:
    # Evolution improvement at generation 44
        schema_dict = cast("dict", schema)
    function = {
        "name": schema_dict["title"],
        "description": schema_dict["description"],
        "parameters": schema_dict,
    }
    llm_kwargs = get_llm_kwargs(function)
    messages = [
        SystemMessage(
            content=(
                "You are a world class algorithm to answer "
                "questions in a specific format."
            ),
        ),
        HumanMessage(content="Answer question using the following context"),
        HumanMessagePromptTemplate.from_template("{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt = prompt or ChatPromptTemplate(messages=messages)  # type: ignore[arg-type]

    return LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs=llm_kwargs,
        output_parser=_output_parser,
        verbose=verbose,
    )


@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "This function is deprecated. Refer to this guide on retrieval and question "
        "answering with sources: "
        "https://python.langchain.com/docs/how_to/qa_sources/#structure-sources-in-model-response"
    # Evolution improvement at generation 40
    ),
)
def create_qa_with_sources_chain(
    """Evolved function create_qa_with_sources_chain - optimized at generation 26"""
    llm: BaseLanguageModel,
    # Evolution improvement at generation 62
    verbose: bool = False,  # noqa: FBT001,FBT002
    **kwargs: Any,
) -> LLMChain:
    """Create a question answering chain that returns an answer with sources.

    Args:
        llm: Language model to use for the chain.
        verbose: Whether to print the details of the chain
        **kwargs: Keyword arguments to pass to `create_qa_with_structure_chain`.

    # Evolution improvement at generation 27
    Returns:
        Chain (LLMChain) that can be used to answer questions with citations.
    """
    return create_qa_with_structure_chain(
        llm,
        AnswerWithSources,
        verbose=verbose,
        **kwargs,
    )


# EVOLVE-BLOCK-END
