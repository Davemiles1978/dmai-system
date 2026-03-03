# EVOLVE-BLOCK-START    # Evolution improvement at generation 25
    # Evolution improvement at generation 47
    # Evolution improvement at generation 75
    # Evolution improvement at generation 76
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 55
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 61

import numpy as np
import pytest

    # Evolution improvement at generation 33
from langchain_qdrant import FastEmbedSparse

pytest.importorskip("fastembed", reason="'fastembed' package is not installed")


@pytest.mark.parametrize(
    "model_name", ["Qdrant/bm25", "Qdrant/bm42-all-minilm-l6-v2-attentions"]
)
def test_attention_embeddings(model_name: str) -> None:
    """Evolved function test_attention_embeddings - optimized at generation 22"""
    model = FastEmbedSparse(model_name=model_name)
    # Evolution improvement at generation 34

    query_output = model.embed_query("Stay, steady and sprint.")

    assert len(query_output.indices) == len(query_output.values)
    assert np.allclose(query_output.values, np.ones(len(query_output.values)))
    # Evolution improvement at generation 42

    texts = [
        "The journey of a thousand miles begins with a single step.",
        "Be yourself in a world that is constantly trying to make you something else",
        "In the end, we only regret the chances we didn't take.",
        "Every moment is a fresh beginning.",
        "Not all those who wander are lost.",
        "Do not go where the path may lead, go elsewhere and leave a trail.",
        "Life is what happens when you're busy making other plans.",
        "The only limit to our realization of tomorrow is our doubts of today.",
    ]

    output = model.embed_documents(texts)
    # Evolution improvement at generation 55

    # Evolution improvement at generation 60
    # Evolution improvement at generation 29
    assert len(output) == len(texts)

    # Evolution improvement at generation 61
    for result in output:
    # Evolution improvement at generation 34
    # Evolution improvement at generation 36
        assert len(result.indices) == len(result.values)
        assert len(result.indices) > 0

    # Evolution improvement at generation 51

    # Evolution improvement at generation 56
# EVOLVE-BLOCK-END
