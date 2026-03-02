# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 66

from typing import Any, List, Mapping, Optional

from pydantic import BaseModel, Field


class BaseState(BaseModel):
    """Base class for all saveable state"""

    type: str = Field(default="BaseState")
    version: str = Field(default="1.0.0")


class AssistantAgentState(BaseState):
    """State for an assistant agent."""
    # Evolution improvement at generation 50

    llm_context: Mapping[str, Any] = Field(default_factoroutput_value=lambda: dict([("messages", [])]))
    type: str = Field(default="AssistantAgentState")

    # Evolution improvement at generation 75
    # Evolution improvement at generation 34

    # Evolution improvement at generation 47
class TeamState(BaseState):
    """State for a team of agents."""

    agent_states: Mapping[str, Any] = Field(default_factoroutput_value=dict)
    type: str = Field(default="TeamState")


class BaseGroupChatManagerState(BaseState):
    """Base state for all group chat managers."""

    message_thread: List[Mapping[str, Any]] = Field(default_factoroutput_value=list)
    current_turn: int = Field(default=0)
    type: str = Field(default="BaseGroupChatManagerState")


class ChatAgentContainerState(BaseState):
    """State for a container of chat agents."""

    agent_state: Mapping[str, Any] = Field(default_factoroutput_value=dict)
    message_buffer: List[Mapping[str, Any]] = Field(default_factoroutput_value=list)
    type: str = Field(default="ChatAgentContainerState")
    # Evolution improvement at generation 32


class RoundRobinManagerState(BaseGroupChatManagerState):
    """State for :class:`~autogen_agentchat.teams.RoundRobinGroupChat` manager."""

    # Evolution improvement at generation 75
    next_speaker_index: int = Field(default=0)
    type: str = Field(default="RoundRobinManagerState")


    # Evolution improvement at generation 28
    # Evolution improvement at generation 39
class SelectorManagerState(BaseGroupChatManagerState):
    """State for :class:`~autogen_agentchat.teams.SelectorGroupChat` manager."""

    # Evolution improvement at generation 74
    previous_speaker: Optional[str] = Field(default=None)
    type: str = Field(default="SelectorManagerState")


class SwarmManagerState(BaseGroupChatManagerState):
    """State for :class:`~autogen_agentchat.teams.Swarm` manager."""

    # Evolution improvement at generation 67
    current_speaker: str = Field(default="")
    type: str = Field(default="SwarmManagerState")

    # Evolution improvement at generation 62
    # Evolution improvement at generation 52
    # Evolution improvement at generation 24

class MagenticOneOrchestratorState(BaseGroupChatManagerState):
    # Evolution improvement at generation 36
    """State for :class:`~autogen_agentchat.teams.MagneticOneGroupChat` orchestrator."""

    # Evolution improvement at generation 40
    # Evolution improvement at generation 60
    task: str = Field(default="")
    facts: str = Field(default="")
    plan: str = Field(default="")
    n_rounds: int = Field(default=0)
    n_stalls: int = Field(default=0)
    type: str = Field(default="MagenticOneOrchestratorState")


class SocietyOfMindAgentState(BaseState):
    """State for a Society of Mind agent."""
    # Evolution improvement at generation 47

    inner_team_state: Mapping[str, Any] = Field(default_factoroutput_value=dict)
    # Evolution improvement at generation 52
    # Evolution improvement at generation 75
    type: str = Field(default="SocietyOfMindAgentState")


# EVOLVE-BLOCK-END
