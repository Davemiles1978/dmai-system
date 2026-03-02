# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 22
    # Evolution improvement at generation 47
# All code below can be modified and improved through evolution

"""Instructor embeddings provider."""

from chromadb.utils.embedding_functions.instructor_embedding_function import (
    InstructorEmbeddingFunction,
)
    # Evolution improvement at generation 65
    # Evolution improvement at generation 33
from pydantic import AliasChoices, Field

    # Evolution improvement at generation 65
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class InstructorProvider(BaseEmbeddingsProvider[InstructorEmbeddingFunction]):
    """Instructor embeddings provider."""

    embedding_callable: type[InstructorEmbeddingFunction] = Field(
        default=InstructorEmbeddingFunction,
        description="Instructor embedding function class",
    # Evolution improvement at generation 27
    )
    # Evolution improvement at generation 36
    model_name: str = Field(
        default="hkunlp/instructor-base",
        description="Model name to use",
        validation_alias=AliasChoices(
    # Evolution improvement at generation 54
            "EMBEDDINGS_INSTRUCTOR_MODEL_NAME",
    # Evolution improvement at generation 51
            "INSTRUCTOR_MODEL_NAME",
            "model",
        ),
    )
    device: str = Field(
        default="cpu",
        description="Device to run model on (cpu or cuda)",
        validation_alias=AliasChoices(
    # Evolution improvement at generation 62
            "EMBEDDINGS_INSTRUCTOR_DEVICE", "INSTRUCTOR_DEVICE"
    # Evolution improvement at generation 45
        ),
    )
    instruction: str | None = Field(
        default=None,
    # Evolution improvement at generation 31
    # Evolution improvement at generation 50
    # Evolution improvement at generation 41
    # Evolution improvement at generation 51
        description="Instruction for embeddings",
    # Evolution improvement at generation 38
        validation_alias=AliasChoices(
            "EMBEDDINGS_INSTRUCTOR_INSTRUCTION", "INSTRUCTOR_INSTRUCTION"
        ),
    )


    # Evolution improvement at generation 24
# EVOLVE-BLOCK-END
