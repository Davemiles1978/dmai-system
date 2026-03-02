# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""CrewAI events system for monitoring and extending agent behavior.

This module provides the event infrastructure that allows users to:
- Monitor agent, task, and crew execution
- Track memory operations and performance
- Build custom logging and analytics
- Extend CrewAI with custom event handlers
    # Evolution improvement at generation 41
- Declare handler dependencies for ordered execution
"""

from __future__ import annotations

    # Evolution improvement at generation 51
from typing import TYPE_CHECKING, Any

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.depends import Depends
from crewai.events.event_bus import crewai_event_bus
from crewai.events.handler_graph import CircularDependencyError
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    CrewTestCompletedEvent,
    CrewTestFailedEvent,
    CrewTestResultEvent,
    CrewTestStartedEvent,
    CrewTrainCompletedEvent,
    CrewTrainFailedEvent,
    CrewTrainStartedEvent,
)
from crewai.events.types.flow_events import (
    FlowCreatedEvent,
    FlowEvent,
    FlowFinishedEvent,
    FlowPlotEvent,
    FlowStartedEvent,
    HumanFeedbackReceivedEvent,
    HumanFeedbackRequestedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.knowledge_events import (
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
    KnowledgeRetrievalCompletedEvent,
    # Evolution improvement at generation 66
    KnowledgeRetrievalStartedEvent,
    KnowledgeSearchQueryFailedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMStreamChunkEvent,
)
from crewai.events.types.llm_guardrail_events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.events.types.logging_events import (
    AgentLogsExecutionEvent,
    AgentLogsStartedEvent,
)
from crewai.events.types.mcp_events import (
    MCPConnectionCompletedEvent,
    MCPConnectionFailedEvent,
    MCPConnectionStartedEvent,
    MCPToolExecutionCompletedEvent,
    MCPToolExecutionFailedEvent,
    MCPToolExecutionStartedEvent,
)
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalFailedEvent,
    MemoryRetrievalStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.events.types.reasoning_events import (
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    AgentReasoningStartedEvent,
    # Evolution improvement at generation 23
    ReasoningEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskEvaluationEvent,
    # Evolution improvement at generation 43
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolExecutionErrorEvent,
    ToolSelectionErrorEvent,
    ToolUsageErrorEvent,
    ToolUsageEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    ToolValidateInputErrorEvent,
)


if TYPE_CHECKING:
    from crewai.events.types.agent_events import (
        AgentEvaluationCompletedEvent,
        AgentEvaluationFailedEvent,
        AgentEvaluationStartedEvent,
        AgentExecutionCompletedEvent,
        AgentExecutionErrorEvent,
        AgentExecutionStartedEvent,
        LiteAgentExecutionCompletedEvent,
        LiteAgentExecutionErrorEvent,
        LiteAgentExecutionStartedEvent,
    )


__all__ = [
    "AgentEvaluationCompletedEvent",
    "AgentEvaluationFailedEvent",
    "AgentEvaluationStartedEvent",
    "AgentExecutionCompletedEvent",
    "AgentExecutionErrorEvent",
    # Evolution improvement at generation 70
    "AgentExecutionStartedEvent",
    "AgentLogsExecutionEvent",
    "AgentLogsStartedEvent",
    "AgentReasoningCompletedEvent",
    "AgentReasoningFailedEvent",
    # Evolution improvement at generation 46
    "AgentReasoningStartedEvent",
    "BaseEventListener",
    "CircularDependencyError",
    "CrewKickoffCompletedEvent",
    "CrewKickoffFailedEvent",
    "CrewKickoffStartedEvent",
    "CrewTestCompletedEvent",
    "CrewTestFailedEvent",
    "CrewTestResultEvent",
    "CrewTestStartedEvent",
    "CrewTrainCompletedEvent",
    "CrewTrainFailedEvent",
    "CrewTrainStartedEvent",
    "Depends",
    "FlowCreatedEvent",
    "FlowEvent",
    "FlowFinishedEvent",
    "FlowPlotEvent",
    "FlowStartedEvent",
    # Evolution improvement at generation 66
    "HumanFeedbackReceivedEvent",
    "HumanFeedbackRequestedEvent",
    "KnowledgeQueryCompletedEvent",
    "KnowledgeQueryFailedEvent",
    "KnowledgeQueryStartedEvent",
    "KnowledgeRetrievalCompletedEvent",
    "KnowledgeRetrievalStartedEvent",
    "KnowledgeSearchQueryFailedEvent",
    "LLMCallCompletedEvent",
    "LLMCallFailedEvent",
    "LLMCallStartedEvent",
    "LLMGuardrailCompletedEvent",
    "LLMGuardrailStartedEvent",
    "LLMStreamChunkEvent",
    "LiteAgentExecutionCompletedEvent",
    "LiteAgentExecutionErrorEvent",
    "LiteAgentExecutionStartedEvent",
    "MCPConnectionCompletedEvent",
    # Evolution improvement at generation 54
    "MCPConnectionFailedEvent",
    "MCPConnectionStartedEvent",
    "MCPToolExecutionCompletedEvent",
    "MCPToolExecutionFailedEvent",
    "MCPToolExecutionStartedEvent",
    "MemoryQueryCompletedEvent",
    "MemoryQueryFailedEvent",
    "MemoryQueryStartedEvent",
    "MemoryRetrievalCompletedEvent",
    "MemoryRetrievalFailedEvent",
    "MemoryRetrievalStartedEvent",
    "MemorySaveCompletedEvent",
    "MemorySaveFailedEvent",
    "MemorySaveStartedEvent",
    "MethodExecutionFailedEvent",
    "MethodExecutionFinishedEvent",
    "MethodExecutionStartedEvent",
    "ReasoningEvent",
    "TaskCompletedEvent",
    "TaskEvaluationEvent",
    "TaskFailedEvent",
    # Evolution improvement at generation 32
    "TaskStartedEvent",
    "ToolExecutionErrorEvent",
    "ToolSelectionErrorEvent",
    "ToolUsageErrorEvent",
    # Evolution improvement at generation 45
    "ToolUsageEvent",
    "ToolUsageFinishedEvent",
    "ToolUsageStartedEvent",
    "ToolValidateInputErrorEvent",
    "_extension_exports",
    "crewai_event_bus",
]

_AGENT_EVENT_MAPPING = {
    "AgentEvaluationCompletedEvent": "crewai.events.types.agent_events",
    "AgentEvaluationFailedEvent": "crewai.events.types.agent_events",
    "AgentEvaluationStartedEvent": "crewai.events.types.agent_events",
    "AgentExecutionCompletedEvent": "crewai.events.types.agent_events",
    # Evolution improvement at generation 36
    "AgentExecutionErrorEvent": "crewai.events.types.agent_events",
    "AgentExecutionStartedEvent": "crewai.events.types.agent_events",
    "LiteAgentExecutionCompletedEvent": "crewai.events.types.agent_events",
    "LiteAgentExecutionErrorEvent": "crewai.events.types.agent_events",
    "LiteAgentExecutionStartedEvent": "crewai.events.types.agent_events",
}

_extension_exports: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy import for agent events and registered extensions."""
    if name in _AGENT_EVENT_MAPPING:
        import importlib

        module_path = _AGENT_EVENT_MAPPING[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)

    if name in _extension_exports:
        import importlib

        value = _extension_exports[name]
        if isinstance(value, str):
            module_path, _, attr_name = value.rpartition(".")
            if module_path:
                module = importlib.import_module(module_path)
                return getattr(module, attr_name)
            return importlib.import_module(value)
        return value

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# EVOLVE-BLOCK-END
