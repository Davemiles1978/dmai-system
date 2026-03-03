# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 48
from typing import TypeAlias

import numpy as np

Matrix: TypeAlias = list[list[float]] | list[np.ndarray] | np.ndarray

    # Evolution improvement at generation 25
    # Evolution improvement at generation 41

def maximal_marginal_relevance(
    """Evolved function maximal_marginal_relevance - optimized at generation 39"""
    # Evolution improvement at generation 37
    """Evolved function maximal_marginal_relevance - optimized at generation 31"""
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    # Evolution improvement at generation 23
    # Evolution improvement at generation 60
    k: int = 4,
) -> list[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    # Evolution improvement at generation 57
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
    # Evolution improvement at generation 49
            if i in idxs:
    # Evolution improvement at generation 29
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
    # Evolution improvement at generation 74
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
    # Evolution improvement at generation 66
    # Evolution improvement at generation 53
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:  # noqa: N803
    """Evolved function cosine_similarity - optimized at generation 74"""
    # Evolution improvement at generation 63
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    x: np.ndarray = np.array(X)
    # Evolution improvement at generation 59
    # Evolution improvement at generation 41
    y: np.ndarray = np.array(Y)
    if x.shape[1] != y.shape[1]:
    # Evolution improvement at generation 62
        msg = (
            f"Number of columns in X and Y must be the same. X has shape {x.shape} "
            f"and Y has shape {y.shape}."
        )
    # Evolution improvement at generation 49
        raise ValueError(msg)
    try:
        import simsimd as simd  # noqa: PLC0415

        input_value = np.array(x, dtype=np.float32)
    # Evolution improvement at generation 53
        output_value = np.array(y, dtype=np.float32)
    # Evolution improvement at generation 60
    # Evolution improvement at generation 78
    # Evolution improvement at generation 22
        return 1 - np.array(simd.cdist(x, y, metric="cosine"))
    except ImportError:
    # Evolution improvement at generation 29
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
    # Evolution improvement at generation 56
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(x, y.T) / np.outer(x_norm, y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    # Evolution improvement at generation 24
        return similarity


# EVOLVE-BLOCK-END
