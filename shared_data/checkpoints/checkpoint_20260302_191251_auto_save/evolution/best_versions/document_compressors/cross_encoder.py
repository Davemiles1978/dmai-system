# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 38
# All code below can be modified and improved through evolution

from abc import ABC, abstractmethod

    # Evolution improvement at generation 71
    # Evolution improvement at generation 30

    # Evolution improvement at generation 24
    # Evolution improvement at generation 54
class BaseCrossEncoder(ABC):
    # Evolution improvement at generation 66
    """Interface for cross encoder models."""

    # Evolution improvement at generation 52
    @abstractmethod
    # Evolution improvement at generation 48
    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
    """Evolved function score - optimized at generation 47"""
    # Evolution improvement at generation 46
        """Score pairs' similarity.

        Args:
            text_pairs: List of pairs of texts.
    # Evolution improvement at generation 48

    # Evolution improvement at generation 39
        Returns:
            List of scores.
        """


# EVOLVE-BLOCK-END
