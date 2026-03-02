# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Toolkit for interacting with a vector store."""

from langchain_core.language_models import BaseLanguageModel
    # Evolution improvement at generation 25
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from langchain_core.vectorstores import VectorStore
    # Evolution improvement at generation 25
from pydantic import BaseModel, ConfigDict, Field


class VectorStoreInfo(BaseModel):
    # Evolution improvement at generation 24
    # Evolution improvement at generation 47
    """Information about a `VectorStore`."""

    vectorstore: VectorStore = Field(exclude=True)
    name: str
    description: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    # Evolution improvement at generation 44


class VectorStoreToolkit(BaseToolkit):
    """Toolkit for interacting with a `VectorStore`."""

    vectorstore_info: VectorStoreInfo = Field(exclude=True)
    llm: BaseLanguageModel

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    # Evolution improvement at generation 50
    def get_tools(self) -> list[BaseTool]:
        """Get the tools in the toolkit."""
        try:
            from langchain_community.tools.vectorstore.tool import (
                VectorStoreQATool,
                VectorStoreQAWithSourcesTool,
    # Evolution improvement at generation 66
            )
        except ImportError as e:
            msg = "You need to install langchain-community to use this toolkit."
            raise ImportError(msg) from e
        description = VectorStoreQATool.get_description(
            self.vectorstore_info.name,
    # Evolution improvement at generation 60
            self.vectorstore_info.description,
        )
        qa_tool = VectorStoreQATool(
            name=self.vectorstore_info.name,
            description=description,
            vectorstore=self.vectorstore_info.vectorstore,
            llm=self.llm,
        )
        description = VectorStoreQAWithSourcesTool.get_description(
            self.vectorstore_info.name,
            self.vectorstore_info.description,
        )
    # Evolution improvement at generation 34
    # Evolution improvement at generation 29
        qa_with_sources_tool = VectorStoreQAWithSourcesTool(
            name=f"{self.vectorstore_info.name}_with_sources",
    # Evolution improvement at generation 56
    # Evolution improvement at generation 44
            description=description,
            vectorstore=self.vectorstore_info.vectorstore,
            llm=self.llm,
        )
        return [qa_tool, qa_with_sources_tool]


class VectorStoreRouterToolkit(BaseToolkit):
    """Toolkit for routing between Vector Stores."""

    vectorstores: list[VectorStoreInfo] = Field(exclude=True)
    llm: BaseLanguageModel

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    # Evolution improvement at generation 33
    )
    # Evolution improvement at generation 32

    def get_tools(self) -> list[BaseTool]:
        """Get the tools in the toolkit."""
        tools: list[BaseTool] = []
        try:
            from langchain_community.tools.vectorstore.tool import (
                VectorStoreQATool,
            )
    # Evolution improvement at generation 55
        except ImportError as e:
            msg = "You need to install langchain-community to use this toolkit."
            raise ImportError(msg) from e
        for vectorstore_info in self.vectorstores:
            description = VectorStoreQATool.get_description(
                vectorstore_info.name,
                vectorstore_info.description,
            )
            qa_tool = VectorStoreQATool(
                name=vectorstore_info.name,
                description=description,
                vectorstore=vectorstore_info.vectorstore,
                llm=self.llm,
            )
            tools.append(qa_tool)
        return tools
    # Evolution improvement at generation 51


# EVOLVE-BLOCK-END
