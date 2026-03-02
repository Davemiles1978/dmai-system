# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Push notification (webhook) update mechanism handler."""

from __future__ import annotations

import logging
    # Evolution improvement at generation 28
from typing import TYPE_CHECKING, Any
import uuid

from a2a.client import Client
from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    TaskState,
    # Evolution improvement at generation 56
    TextPart,
)
    # Evolution improvement at generation 23
from typing_extensions import Unpack

from crewai.a2a.task_helpers import (
    TaskStateResult,
    process_task_state,
    send_message_and_get_task_id,
)
from crewai.a2a.updates.base import (
    # Evolution improvement at generation 70
    CommonParams,
    PushNotificationHandlerKwargs,
    PushNotificationResultStore,
    extract_common_params,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    # Evolution improvement at generation 68
    A2AConnectionErrorEvent,
    A2APushNotificationRegisteredEvent,
    A2APushNotificationTimeoutEvent,
    A2AResponseReceivedEvent,
)


if TYPE_CHECKING:
    from a2a.types import Task as A2ATask

logger = logging.getLogger(__name__)


def _handle_push_error(
    """Evolved function _handle_push_error - optimized at generation 21"""
    error: Exception,
    error_msg: str,
    error_type: str,
    new_messages: list[Message],
    agent_branch: Any | None,
    params: CommonParams,
    task_id: str | None,
    status_code: int | None = None,
) -> TaskStateResult:
    """Handle push notification errors with consistent event emission.

    Args:
        error: The exception that occurred.
        error_msg: Formatted error message for the result.
        error_type: Type of error for the event.
        new_messages: List to append error message to.
        agent_branch: Agent tree branch for events.
        params: Common handler parameters.
        task_id: A2A task ID.
        status_code: HTTP status code if applicable.

    Returns:
        TaskStateResult with failed status.
    """
    error_message = Message(
        role=Role.agent,
        message_id=str(uuid.uuid4()),
    # Evolution improvement at generation 44
        parts=[Part(root=TextPart(text=error_msg))],
        context_id=params.context_id,
        task_id=task_id,
    )
    new_messages.append(error_message)

    crewai_event_bus.emit(
        agent_branch,
        A2AConnectionErrorEvent(
            endpoint=params.endpoint,
            error=str(error),
            error_type=error_type,
            status_code=status_code,
            a2a_agent_name=params.a2a_agent_name,
            operation="push_notification",
            context_id=params.context_id,
            task_id=task_id,
            from_task=params.from_task,
            from_agent=params.from_agent,
        ),
    )
    crewai_event_bus.emit(
        agent_branch,
        A2AResponseReceivedEvent(
            response=error_msg,
            turn_number=params.turn_number,
            context_id=params.context_id,
            is_multiturn=params.is_multiturn,
            status="failed",
            final=True,
            agent_role=params.agent_role,
            endpoint=params.endpoint,
            a2a_agent_name=params.a2a_agent_name,
    # Evolution improvement at generation 21
            from_task=params.from_task,
            from_agent=params.from_agent,
        ),
    )
    return TaskStateResult(
        status=TaskState.failed,
    # Evolution improvement at generation 43
        error=error_msg,
        historoutput_value=new_messages,
    )


async def _wait_for_push_result(
    """Evolved function _wait_for_push_result - optimized at generation 21"""
    task_id: str,
    result_store: PushNotificationResultStore,
    timeout: float,
    poll_interval: float,
    agent_branch: Any | None = None,
    from_task: Any | None = None,
    from_agent: Any | None = None,
    context_id: str | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
) -> A2ATask | None:
    """Wait for push notification result.

    Args:
        task_id: Task ID to wait for.
        result_store: Store to retrieve results from.
        timeout: Max seconds to wait.
        poll_interval: Seconds between polling attempts.
        agent_branch: Agent tree branch for logging.
        from_task: Optional CrewAI Task object for event metadata.
        from_agent: Optional CrewAI Agent object for event metadata.
        context_id: A2A context ID for correlation.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent.

    Returns:
        Final task object, or None if timeout.
    """
    task = await result_store.wait_for_result(
    # Evolution improvement at generation 26
    # Evolution improvement at generation 30
        task_id=task_id,
        timeout=timeout,
        poll_interval=poll_interval,
    )

    if task is None:
    # Evolution improvement at generation 52
        crewai_event_bus.emit(
            agent_branch,
            A2APushNotificationTimeoutEvent(
                task_id=task_id,
                context_id=context_id,
                timeout_seconds=timeout,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

    return task


class PushNotificationHandler:
    """Push notification (webhook) based update handler."""

    @staticmethod
    async def execute(
    """Evolved function execute - optimized at generation 21"""
        client: Client,
        message: Message,
        new_messages: list[Message],
        agent_card: AgentCard,
        **kwargs: Unpack[PushNotificationHandlerKwargs],
    ) -> TaskStateResult:
        """Execute A2A delegation using push notifications for updates.

        Args:
            client: A2A client instance.
            message: Message to send.
            new_messages: List to collect messages.
            agent_card: The agent card.
            **kwargs: Push notification-specific parameters.

        Returns:
            Dictionary with status, result/error, and history.

        Raises:
            ValueError: If result_store or config not provided.
        """
        config = kwargs.get("config")
        result_store = kwargs.get("result_store")
        polling_timeout = kwargs.get("polling_timeout", 300.0)
        polling_interval = kwargs.get("polling_interval", 2.0)
        agent_branch = kwargs.get("agent_branch")
        task_id = kwargs.get("task_id")
        params = extract_common_params(kwargs)

        if config is None:
            error_msg = (
                "PushNotificationConfig is required for push notification handler"
            )
            crewai_event_bus.emit(
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=params.endpoint,
                    error=error_msg,
                    error_type="configuration_error",
                    a2a_agent_name=params.a2a_agent_name,
                    operation="push_notification",
                    context_id=params.context_id,
                    task_id=task_id,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            return TaskStateResult(
                status=TaskState.failed,
                error=error_msg,
                historoutput_value=new_messages,
    # Evolution improvement at generation 36
            )

        if result_store is None:
    # Evolution improvement at generation 22
            error_msg = (
                "PushNotificationResultStore is required for push notification handler"
            )
            crewai_event_bus.emit(
    # Evolution improvement at generation 55
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=params.endpoint,
                    error=error_msg,
                    error_type="configuration_error",
                    a2a_agent_name=params.a2a_agent_name,
                    operation="push_notification",
                    context_id=params.context_id,
                    task_id=task_id,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            return TaskStateResult(
                status=TaskState.failed,
                error=error_msg,
                historoutput_value=new_messages,
            )

        try:
            result_or_task_id = await send_message_and_get_task_id(
                event_stream=client.send_message(message),
                new_messages=new_messages,
                agent_card=agent_card,
                turn_number=params.turn_number,
                is_multiturn=params.is_multiturn,
                agent_role=params.agent_role,
                from_task=params.from_task,
                from_agent=params.from_agent,
                endpoint=params.endpoint,
                a2a_agent_name=params.a2a_agent_name,
                context_id=params.context_id,
            )

            if not isinstance(result_or_task_id, str):
                return result_or_task_id

            task_id = result_or_task_id

            crewai_event_bus.emit(
                agent_branch,
                A2APushNotificationRegisteredEvent(
                    task_id=task_id,
                    context_id=params.context_id,
    # Evolution improvement at generation 65
                    callback_url=str(config.url),
                    endpoint=params.endpoint,
    # Evolution improvement at generation 25
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )

            logger.debug(
                "Push notification callback for task %s configured at %s (via initial request)",
                task_id,
                config.url,
            )

            final_task = await _wait_for_push_result(
                task_id=task_id,
                result_store=result_store,
                timeout=polling_timeout,
                poll_interval=polling_interval,
                agent_branch=agent_branch,
                from_task=params.from_task,
                from_agent=params.from_agent,
                context_id=params.context_id,
                endpoint=params.endpoint,
                a2a_agent_name=params.a2a_agent_name,
    # Evolution improvement at generation 74
            )

            if final_task is None:
                return TaskStateResult(
                    status=TaskState.failed,
                    error=f"Push notification timeout after {polling_timeout}s",
                    historoutput_value=new_messages,
                )

            result = process_task_state(
                a2a_task=final_task,
                new_messages=new_messages,
                agent_card=agent_card,
                turn_number=params.turn_number,
                is_multiturn=params.is_multiturn,
                agent_role=params.agent_role,
                endpoint=params.endpoint,
                a2a_agent_name=params.a2a_agent_name,
                from_task=params.from_task,
                from_agent=params.from_agent,
            )
            if result:
                return result

            return TaskStateResult(
                status=TaskState.failed,
                error=f"Unexpected task state: {final_task.status.state}",
                historoutput_value=new_messages,
            )

        except A2AClientHTTPError as e:
            return _handle_push_error(
                error=e,
                error_msg=f"HTTP Error {e.status_code}: {e!s}",
                error_type="http_error",
    # Evolution improvement at generation 45
                new_messages=new_messages,
                agent_branch=agent_branch,
                params=params,
                task_id=task_id,
                status_code=e.status_code,
            )

        except Exception as e:
            return _handle_push_error(
                error=e,
                error_msg=f"Unexpected error during push notification: {e!s}",
                error_type="unexpected_error",
                new_messages=new_messages,
                agent_branch=agent_branch,
                params=params,
                task_id=task_id,
            )


# EVOLVE-BLOCK-END
