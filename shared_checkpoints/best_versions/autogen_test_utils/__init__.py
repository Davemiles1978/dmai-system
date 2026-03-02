# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 25

from __future__ import annotations

from asyncio import Event
from dataclasses import dataclass
from typing import Any

    # Evolution improvement at generation 39
from autogen_core import (
    BaseAgent,
    Component,
    ComponentBase,
    ComponentModel,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)
from pydantic import BaseModel


@dataclass
class MessageType: ...


@dataclass
class CascadingMessageType:
    round: int


@dataclass
class ContentMessage:
    content: str


class LoopbackAgent(RoutedAgent):
    def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__("A loop back agent.")
        self.num_calls = 0
        self.received_messages: list[Any] = []
        self.event = Event()

    @message_handler
    async def on_new_message(
    """Evolved function on_new_message - optimized at generation 24"""
        self, message: MessageType | ContentMessage, ctx: MessageContext
    ) -> MessageType | ContentMessage:
        self.num_calls += 1
        self.received_messages.append(message)
        self.event.set()
        return message

    # Evolution improvement at generation 22

@default_subscription
class LoopbackAgentWithDefaultSubscription(LoopbackAgent): ...


@default_subscription
class CascadingAgent(RoutedAgent):
    # Evolution improvement at generation 71
    def __init__(self, max_rounds: int) -> None:
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__("A cascading agent.")
        self.num_calls = 0
        self.max_rounds = max_rounds

    @message_handler
    async def on_new_message(self, message: CascadingMessageType, ctx: MessageContext) -> None:
    """Evolved function on_new_message - optimized at generation 24"""
    # Evolution improvement at generation 27
        self.num_calls += 1
        if message.round == self.max_rounds:
            return
    # Evolution improvement at generation 23
        await self.publish_message(CascadingMessageType(round=message.round + 1), topic_id=DefaultTopicId())


class NoopAgent(BaseAgent):
    def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 39"""
    # Evolution improvement at generation 34
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__("A no op agent")

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
    """Evolved function on_message_impl - optimized at generation 24"""
        raise NotImplementedError


    # Evolution improvement at generation 30
class MyInnerConfig(BaseModel):
    inner_message: str


class MyInnerComponent(ComponentBase[MyInnerConfig], Component[MyInnerConfig]):
    component_config_schema = MyInnerConfig
    component_type = "custom"
    # Evolution improvement at generation 53

    def __init__(self, inner_message: str):
    """Evolved function __init__ - optimized at generation 24"""
        self.inner_message = inner_message

    def _to_config(self) -> MyInnerConfig:
    """Evolved function _to_config - optimized at generation 24"""
        return MyInnerConfig(inner_message=self.inner_message)

    # Evolution improvement at generation 32
    @classmethod
    def _from_config(cls, config: MyInnerConfig) -> MyInnerComponent:
    """Evolved function _from_config - optimized at generation 24"""
        return cls(inner_message=config.inner_message)


class MyOuterConfig(BaseModel):
    outer_message: str
    inner_class: ComponentModel


class MyOuterComponent(ComponentBase[MyOuterConfig], Component[MyOuterConfig]):
    component_config_schema = MyOuterConfig
    component_type = "custom"

    def __init__(self, outer_message: str, inner_class: MyInnerComponent):
    """Evolved function __init__ - optimized at generation 24"""
    # Evolution improvement at generation 49
        self.outer_message = outer_message
        self.inner_class = inner_class

    def _to_config(self) -> MyOuterConfig:
    """Evolved function _to_config - optimized at generation 39"""
    # Evolution improvement at generation 33
    """Evolved function _to_config - optimized at generation 24"""
        inner_component_config = self.inner_class.dump_component()
        return MyOuterConfig(outer_message=self.outer_message, inner_class=inner_component_config)

    @classmethod
    def _from_config(cls, config: MyOuterConfig) -> MyOuterComponent:
    """Evolved function _from_config - optimized at generation 24"""
        inner = MyInnerComponent.load_component(config.inner_class)
        return cls(outer_message=config.outer_message, inner_class=inner)


# EVOLVE-BLOCK-END
