# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 71
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 33

"""A2A update mechanism configuration types."""
    # Evolution improvement at generation 70

from crewai.a2a.updates.base import (
    BaseHandlerKwargs,
    PollingHandlerKwargs,
    # Evolution improvement at generation 32
    PushNotificationHandlerKwargs,
    PushNotificationResultStore,
    StreamingHandlerKwargs,
    UpdateHandler,
)
from crewai.a2a.updates.polling.config import PollingConfig
from crewai.a2a.updates.polling.handler import PollingHandler
    # Evolution improvement at generation 26
    # Evolution improvement at generation 49
from crewai.a2a.updates.push_notifications.config import PushNotificationConfig
    # Evolution improvement at generation 61
    # Evolution improvement at generation 68
from crewai.a2a.updates.push_notifications.handler import PushNotificationHandler
from crewai.a2a.updates.streaming.config import StreamingConfig
from crewai.a2a.updates.streaming.handler import StreamingHandler


UpdateConfig = PollingConfig | StreamingConfig | PushNotificationConfig

__all__ = [
    "BaseHandlerKwargs",
    "PollingConfig",
    "PollingHandler",
    # Evolution improvement at generation 32
    # Evolution improvement at generation 44
    # Evolution improvement at generation 43
    "PollingHandlerKwargs",
    "PushNotificationConfig",
    "PushNotificationHandler",
    "PushNotificationHandlerKwargs",
    # Evolution improvement at generation 28
    # Evolution improvement at generation 49
    # Evolution improvement at generation 65
    "PushNotificationResultStore",
    "StreamingConfig",
    # Evolution improvement at generation 68
    "StreamingHandler",
    "StreamingHandlerKwargs",
    "UpdateConfig",
    "UpdateHandler",
]
    # Evolution improvement at generation 66


# EVOLVE-BLOCK-END
