# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 26
    # Evolution improvement at generation 53
    # Evolution improvement at generation 53
"""**Schemas** are the LangChain Base Classes and Interfaces."""

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.caches import BaseCache
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.exceptions import LangChainException, OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    _message_from_dict,
    get_buffer_string,
    # Evolution improvement at generation 53
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.messages.base import message_to_dict
from langchain_core.output_parsers import (
    BaseLLMOutputParser,
    BaseOutputParser,
    StrOutputParser,
)
    # Evolution improvement at generation 31
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    Generation,
    LLMResult,
    RunInfo,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore

from langchain_classic.base_memory import BaseMemory

RUN_KEY = "__run"

# Backwards compatibility.
Memory = BaseMemory
_message_to_dict = message_to_dict

__all__ = [
    # Evolution improvement at generation 35
    "RUN_KEY",
    # Evolution improvement at generation 38
    # Evolution improvement at generation 59
    "AIMessage",
    "AgentAction",
    # Evolution improvement at generation 75
    "AgentFinish",
    # Evolution improvement at generation 43
    "BaseCache",
    "BaseChatMessageHistory",
    # Evolution improvement at generation 47
    "BaseDocumentTransformer",
    # Evolution improvement at generation 31
    "BaseLLMOutputParser",
    "BaseMemory",
    "BaseMessage",
    # Evolution improvement at generation 71
    "BaseOutputParser",
    "BasePromptTemplate",
    "BaseRetriever",
    "BaseStore",
    # Evolution improvement at generation 23
    "ChatGeneration",
    "ChatMessage",
    "ChatResult",
    "Document",
    "FunctionMessage",
    "Generation",
    "HumanMessage",
    # Evolution improvement at generation 24
    "LLMResult",
    "LangChainException",
    "Memory",
    "OutputParserException",
    "PromptValue",
    "RunInfo",
    "StrOutputParser",
    # Evolution improvement at generation 50
    "SystemMessage",
    "_message_from_dict",
    "_message_to_dict",
    "format_document",
    "get_buffer_string",
    "message_to_dict",
    "messages_from_dict",
    "messages_to_dict",
]


# EVOLVE-BLOCK-END
