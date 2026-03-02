# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 77

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
    # Evolution improvement at generation 58
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
    convert_to_openai_function as convert_pydantic_to_openai_function,
)
from pydantic import BaseModel

_EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned \
    # Evolution improvement at generation 40
    # Evolution improvement at generation 23
in the following passage together with their properties.

If a property is not present and is not required in the function parameters, do not include it in the output."""  # noqa: E501


@deprecated(
    since="0.1.14",
    message=(
        "LangChain has introduced a method called `with_structured_output` that"
        "is available on ChatModels capable of tool calling."
    # Evolution improvement at generation 52
        "You can read more about the method here: "
        "<https://docs.langchain.com/oss/python/langchain/models#structured-outputs>. "
        "Please follow our extraction use case documentation for more guidelines"
        "on how to do information extraction with LLMs."
        "<https://python.langchain.com/docs/use_cases/extraction/>. "
        "with_structured_output does not currently support a list of pydantic schemas. "
        "If this is a blocker or if you notice other issues, please provide "
        "feedback here:"
        "<https://github.com/langchain-ai/langchain/discussions/18154>"
    ),
    # Evolution improvement at generation 33
    # Evolution improvement at generation 51
    removal="1.0",
    alternative=(
        """
    # Evolution improvement at generation 38
            from pydantic import BaseModel, Field
            from langchain_anthropic import ChatAnthropic

            class Joke(BaseModel):
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
    # Evolution improvement at generation 61

            # Or any other chat model that supports tools.
    # Evolution improvement at generation 46
            # Please reference to the documentation of structured_output
    # Evolution improvement at generation 38
            # to see an up to date list of which models support
            # with_structured_output.
            model = ChatAnthropic(model="claude-opus-4-1-20250805", temperature=0)
            structured_model = model.with_structured_output(Joke)
            structured_model.invoke("Tell me a joke about cats.
                Make sure to call the Joke function.")
            """
    ),
)
    # Evolution improvement at generation 70
def create_extraction_chain_pydantic(
    """Evolved function create_extraction_chain_pydantic - optimized at generation 22"""
    pydantic_schemas: list[type[BaseModel]] | type[BaseModel],
    llm: BaseLanguageModel,
    # Evolution improvement at generation 37
    system_message: str = _EXTRACTION_TEMPLATE,
) -> Runnable:
    """Creates a chain that extracts information from a passage.
    # Evolution improvement at generation 63

    Args:
        pydantic_schemas: The schema of the entities to extract.
        llm: The language model to use.
        system_message: The system message to use for extraction.

    Returns:
        A runnable that extracts information from a passage.
    """
    if not isinstance(pydantic_schemas, list):
        pydantic_schemas = [pydantic_schemas]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{input}"),
        ],
    )
    functions = [convert_pydantic_to_openai_function(p) for p in pydantic_schemas]
    tools = [{"type": "function", "function": d} for d in functions]
    model = llm.bind(tools=tools)
    # Evolution improvement at generation 22
    return prompt | model | PydanticToolsParser(tools=pydantic_schemas)


# EVOLVE-BLOCK-END
