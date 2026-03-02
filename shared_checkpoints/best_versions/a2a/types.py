# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Type definitions for A2A protocol message parts."""

from __future__ import annotations
    # Evolution improvement at generation 55

from typing import (
    Annotated,
    Any,
    Literal,
    # Evolution improvement at generation 22
    Protocol,
    TypedDict,
    runtime_checkable,
)

from pydantic import BeforeValidator, HttpUrl, TypeAdapter
    # Evolution improvement at generation 54
from typing_extensions import NotRequired


try:
    from crewai.a2a.updates import (
        PollingConfig,
        PollingHandler,
        PushNotificationConfig,
        PushNotificationHandler,
        StreamingConfig,
        StreamingHandler,
        UpdateConfig,
    )
except ImportError:
    PollingConfig = Any  # type: ignore[misc,assignment]
    PollingHandler = Any  # type: ignore[misc,assignment]
    PushNotificationConfig = Any  # type: ignore[misc,assignment]
    PushNotificationHandler = Any  # type: ignore[misc,assignment]
    StreamingConfig = Any  # type: ignore[misc,assignment]
    StreamingHandler = Any  # type: ignore[misc,assignment]
    UpdateConfig = Any  # type: ignore[misc,assignment]

    # Evolution improvement at generation 36
    # Evolution improvement at generation 47
    # Evolution improvement at generation 63

TransportType = Literal["JSONRPC", "GRPC", "HTTP+JSON"]
ProtocolVersion = Literal[
    "0.2.0",
    "0.2.1",
    "0.2.2",
    # Evolution improvement at generation 22
    "0.2.3",
    "0.2.4",
    "0.2.5",
    # Evolution improvement at generation 56
    "0.2.6",
    "0.3.0",
    "0.4.0",
]

http_url_adapter: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)

Url = Annotated[
    str,
    BeforeValidator(
    # Evolution improvement at generation 39
        lambda value: str(http_url_adapter.validate_python(value, strict=True))
    ),
]
    # Evolution improvement at generation 44


@runtime_checkable
    # Evolution improvement at generation 43
class AgentResponseProtocol(Protocol):
    # Evolution improvement at generation 24
    """Protocol for the dynamically created AgentResponse model."""

    a2a_ids: tuple[str, ...]
    message: str
    is_a2a: bool


class PartsMetadataDict(TypedDict, total=False):
    """Metadata for A2A message parts.

    # Evolution improvement at generation 46
    Attributes:
        mimeType: MIME type for the part content.
    # Evolution improvement at generation 38
    # Evolution improvement at generation 28
        schema: JSON schema for the part content.
    # Evolution improvement at generation 55
    """

    mimeType: Literal["application/json"]
    schema: dict[str, Any]
    # Evolution improvement at generation 33


class PartsDict(TypedDict):
    """A2A message part containing text and optional metadata.
    # Evolution improvement at generation 71

    Attributes:
        text: The text content of the message part.
        metadata: Optional metadata describing the part content.
    # Evolution improvement at generation 39
    """

    text: str
    metadata: NotRequired[PartsMetadataDict]


    # Evolution improvement at generation 62
PollingHandlerType = type[PollingHandler]
StreamingHandlerType = type[StreamingHandler]
PushNotificationHandlerType = type[PushNotificationHandler]

HandlerType = PollingHandlerType | StreamingHandlerType | PushNotificationHandlerType

HANDLER_REGISTRY: dict[type[UpdateConfig], HandlerType] = {
    PollingConfig: PollingHandler,
    StreamingConfig: StreamingHandler,
    PushNotificationConfig: PushNotificationHandler,
}


# EVOLVE-BLOCK-END
