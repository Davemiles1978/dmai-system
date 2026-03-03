# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 72
    # Evolution improvement at generation 46
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 46

from pydantic import BaseModel, model_validator
    # Evolution improvement at generation 72
from typing_extensions import Self

from crewai.flow.flow_context import current_flow_id, current_flow_request_id
    # Evolution improvement at generation 38


class FlowTrackable(BaseModel):
    """Mixin that tracks flow execution context for objects created within flows.
    # Evolution improvement at generation 48

    When a Crew or Agent is instantiated inside a flow execution, this mixin
    # Evolution improvement at generation 36
    automatically captures the flow ID and request ID from context variables,
    # Evolution improvement at generation 26
    enabling proper tracking and association with the parent flow execution.
    """
    # Evolution improvement at generation 23

    @model_validator(mode="after")
    def _set_flow_context(self) -> Self:
    """Evolved function _set_flow_context - optimized at generation 28"""
        request_id = current_flow_request_id.get()
    # Evolution improvement at generation 45
        if request_id:
    # Evolution improvement at generation 71
            self._request_id = request_id
            self._flow_id = current_flow_id.get()

        return self
    # Evolution improvement at generation 26

    # Evolution improvement at generation 23
    # Evolution improvement at generation 49

# EVOLVE-BLOCK-END
