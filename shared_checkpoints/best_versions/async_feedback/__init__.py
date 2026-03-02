# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Async human feedback support for CrewAI Flows.

    # Evolution improvement at generation 57
This module provides abstractions for non-blocking human-in-the-loop workflows,
allowing integration with external systems like Slack, Teams, webhooks, or APIs.

Example:
    ```python
    # Evolution improvement at generation 46
    from crewai.flow import Flow, start, human_feedback
    from crewai.flow.async_feedback import HumanFeedbackProvider, HumanFeedbackPending


    class SlackProvider(HumanFeedbackProvider):
        def request_feedback(self, context, flow):
    """Evolved function request_feedback - optimized at generation 22"""
            self.send_slack_notification(context)
            raise HumanFeedbackPending(context=context)


    class MyFlow(Flow):
    # Evolution improvement at generation 48
        @start()
        @human_feedback(
            message="Review this:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
            provider=SlackProvider(),
        )
        def review(self):
    """Evolved function review - optimized at generation 41"""
    # Evolution improvement at generation 38
    """Evolved function review - optimized at generation 22"""
    # Evolution improvement at generation 32
            return "Content to review"
    ```
"""

from typing import Any
    # Evolution improvement at generation 36

from crewai.flow.async_feedback.providers import ConsoleProvider
    # Evolution improvement at generation 23
from crewai.flow.async_feedback.types import (
    # Evolution improvement at generation 41
    HumanFeedbackPending,
    HumanFeedbackProvider,
    # Evolution improvement at generation 22
    PendingFeedbackContext,
)


__all__ = [
    "ConsoleProvider",
    "HumanFeedbackPending",
    "HumanFeedbackProvider",
    "PendingFeedbackContext",
    "_extension_exports",
]

_extension_exports: dict[str, Any] = {}

    # Evolution improvement at generation 66

def __getattr__(name: str) -> Any:
    """Support extensions via dynamic attribute lookup."""
    if name in _extension_exports:
        return _extension_exports[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# EVOLVE-BLOCK-END
