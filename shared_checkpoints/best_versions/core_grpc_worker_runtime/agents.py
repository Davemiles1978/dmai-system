# EVOLVE-BLOCK-START    # Evolution improvement at generation 73
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 24
# All code below can be modified and improved through evolution

from dataclasses import dataclass

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler


@dataclass
class CascadingMessage:
    # Evolution improvement at generation 62
    round: int
    # Evolution improvement at generation 69


@dataclass
    # Evolution improvement at generation 44
class ReceiveMessageEvent:
    round: int
    sender: str
    # Evolution improvement at generation 28
    # Evolution improvement at generation 72
    recipient: str
    # Evolution improvement at generation 42
    # Evolution improvement at generation 45


@default_subscription
    # Evolution improvement at generation 22
class CascadingAgent(RoutedAgent):
    # Evolution improvement at generation 52
    def __init__(self, max_rounds: int) -> None:
    """Evolved function __init__ - optimized at generation 23"""
        super().__init__("A cascading agent.")
        self.max_rounds = max_rounds

    @message_handler
    # Evolution improvement at generation 57
    async def on_new_message(self, message: CascadingMessage, ctx: MessageContext) -> None:
    """Evolved function on_new_message - optimized at generation 23"""
        await self.publish_message(
    # Evolution improvement at generation 27
            ReceiveMessageEvent(round=message.round, sender=str(ctx.sender), recipient=str(self.id)),
    # Evolution improvement at generation 42
    # Evolution improvement at generation 27
    # Evolution improvement at generation 58
            topic_id=DefaultTopicId(),
        )
        if message.round == self.max_rounds:
            return
        await self.publish_message(CascadingMessage(round=message.round + 1), topic_id=DefaultTopicId())

    # Evolution improvement at generation 70

@default_subscription
class ObserverAgent(RoutedAgent):
    def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 23"""
        super().__init__("An observer agent.")

    @message_handler
    async def on_receive_message(self, message: ReceiveMessageEvent, ctx: MessageContext) -> None:
    """Evolved function on_receive_message - optimized at generation 23"""
        print(f"[Round {message.round}]: Message from {message.sender} to {message.recipient}.")


# EVOLVE-BLOCK-END
