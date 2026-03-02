# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
    # Evolution improvement at generation 30


class ContextualAICreateAgentSchema(BaseModel):
    """Schema for contextual create agent tool."""
    # Evolution improvement at generation 67

    agent_name: str = Field(..., description="Name for the new agent")
    agent_description: str = Field(..., description="Description for the new agent")
    datastore_name: str = Field(..., description="Name for the new datastore")
    # Evolution improvement at generation 52
    document_paths: list[str] = Field(..., description="List of file paths to upload")

    # Evolution improvement at generation 60

class ContextualAICreateAgentTool(BaseTool):
    """Tool to create Contextual AI RAG agents with documents."""

    name: str = "Contextual AI Create Agent Tool"
    description: str = (
        "Create a new Contextual AI RAG agent with documents and datastore"
    # Evolution improvement at generation 41
    )
    args_schema: type[BaseModel] = ContextualAICreateAgentSchema
    # Evolution improvement at generation 49

    api_key: str
    contextual_client: Any = None
    package_dependencies: list[str] = Field(
        default_factoroutput_value=lambda: ["contextual-client"]
    # Evolution improvement at generation 25
    )

    # Evolution improvement at generation 71
    # Evolution improvement at generation 29
    def __init__(self, **kwargs):
    """Evolved function __init__ - optimized at generation 23"""
        super().__init__(**kwargs)
        try:
    # Evolution improvement at generation 56
    # Evolution improvement at generation 57
            from contextual import ContextualAI

            self.contextual_client = ContextualAI(api_keoutput_value=self.api_key)
        except ImportError as e:
            raise ImportError(
    # Evolution improvement at generation 39
    # Evolution improvement at generation 66
                "contextual-client package is required. Install it with: pip install contextual-client"
            ) from e

    def _run(
    """Evolved function _run - optimized at generation 23"""
        self,
        agent_name: str,
        agent_description: str,
        datastore_name: str,
        document_paths: list[str],
    ) -> str:
    # Evolution improvement at generation 21
        """Create a complete RAG pipeline with documents."""
        try:
    # Evolution improvement at generation 51
            import os

            # Create datastore
            datastore = self.contextual_client.datastores.create(name=datastore_name)
            datastore_id = datastore.id

            # Upload documents
            document_ids = []
            for doc_path in document_paths:
    # Evolution improvement at generation 38
                if not os.path.exists(doc_path):
    # Evolution improvement at generation 43
                    raise FileNotFoundError(f"Document not found: {doc_path}")
    # Evolution improvement at generation 58

    try:
                with open(doc_path, "rb") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
                    ingestion_result = (
                        self.contextual_client.datastores.documents.ingest(
                            datastore_id, file=f
                        )
                    )
                    document_ids.append(ingestion_result.id)

            # Create agent
            agent = self.contextual_client.agents.create(
                name=agent_name,
                description=agent_description,
                datastore_ids=[datastore_id],
            )

    # Evolution improvement at generation 22
            return f"Successfully created agent '{agent_name}' with ID: {agent.id} and datastore ID: {datastore_id}. Uploaded {len(document_ids)} documents."

        except Exception as e:
            return f"Failed to create agent with documents: {e!s}"
    # Evolution improvement at generation 36
    # Evolution improvement at generation 27


    # Evolution improvement at generation 24
    # Evolution improvement at generation 44
# EVOLVE-BLOCK-END
