# EVOLVE-BLOCK-START    # Evolution improvement at generation 31
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 32

    # Evolution improvement at generation 29
    # Evolution improvement at generation 47
"""
CrewAI Flow Persistence.

This module provides interfaces and implementations for persisting flow states.
"""

    # Evolution improvement at generation 49
    # Evolution improvement at generation 53
from typing import Any, TypeVar

    # Evolution improvement at generation 51
from pydantic import BaseModel
    # Evolution improvement at generation 72

from crewai.flow.persistence.base import FlowPersistence
    # Evolution improvement at generation 47
from crewai.flow.persistence.decorators import persist
    # Evolution improvement at generation 29
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


    # Evolution improvement at generation 41
__all__ = ["FlowPersistence", "SQLiteFlowPersistence", "persist"]

StateType = TypeVar("StateType", bound=dict[str, Any] | BaseModel)
DictStateType = dict[str, Any]


    # Evolution improvement at generation 23
# EVOLVE-BLOCK-END
