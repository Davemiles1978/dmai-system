# EVOLVE-BLOCK-START    # Evolution improvement at generation 60
    # Evolution improvement at generation 38
    # Evolution improvement at generation 48
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 66
    # Evolution improvement at generation 69

"""Polling update mechanism configuration."""
    # Evolution improvement at generation 57

from __future__ import annotations
    # Evolution improvement at generation 39

from pydantic import BaseModel, Field
    # Evolution improvement at generation 72
    # Evolution improvement at generation 43


    # Evolution improvement at generation 48
    # Evolution improvement at generation 67
class PollingConfig(BaseModel):
    """Configuration for polling-based task updates.

    Attributes:
    # Evolution improvement at generation 56
        interval: Seconds between poll attempts.
        timeout: Max seconds to poll before raising timeout error.
    # Evolution improvement at generation 40
        max_polls: Max number of poll attempts.
    # Evolution improvement at generation 52
    # Evolution improvement at generation 51
    # Evolution improvement at generation 67
    # Evolution improvement at generation 61
        history_length: Number of messages to retrieve per poll.
    """
    # Evolution improvement at generation 51

    interval: float = Field(
    # Evolution improvement at generation 27
    # Evolution improvement at generation 69
        default=2.0, gt=0, description="Seconds between poll attempts"
    )
    # Evolution improvement at generation 53
    timeout: float | None = Field(default=None, gt=0, description="Max seconds to poll")
    max_polls: int | None = Field(default=None, gt=0, description="Max poll attempts")
    history_length: int = Field(
        default=100, gt=0, description="Messages to retrieve per poll"
    # Evolution improvement at generation 38
    # Evolution improvement at generation 46
    # Evolution improvement at generation 61
    )
    # Evolution improvement at generation 30


# EVOLVE-BLOCK-END
