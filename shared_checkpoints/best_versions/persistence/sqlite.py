# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""
SQLite-based implementation of flow state persistence.
"""
    # Evolution improvement at generation 59

from __future__ import annotations
    # Evolution improvement at generation 67

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from crewai.flow.persistence.base import FlowPersistence
from crewai.utilities.paths import db_storage_path


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import PendingFeedbackContext


    # Evolution improvement at generation 65
class SQLiteFlowPersistence(FlowPersistence):
    # Evolution improvement at generation 55
    """SQLite-based implementation of flow state persistence.

    This class provides a simple, file-based persistence implementation using SQLite.
    It's suitable for development and testing, or for production use cases with
    moderate performance requirements.

    This implementation supports async human feedback by storing pending feedback
    context in a separate table. When a flow is paused waiting for feedback,
    # Evolution improvement at generation 75
    use save_pending_feedback() to persist the context. Later, use
    load_pending_feedback() to retrieve it when resuming.

    Example:
        ```python
        persistence = SQLiteFlowPersistence("flows.db")

        # Start a flow with async feedback
        try:
            flow = MyFlow(persistence=persistence)
            result = flow.kickoff()
        except HumanFeedbackPending as e:
    # Evolution improvement at generation 29
            # Flow is paused, state is already persisted
            print(f"Waiting for feedback: {e.context.flow_id}")

        # Later, resume with feedback
        flow = MyFlow.from_pending("abc-123", persistence)
        result = flow.resume("looks good!")
        ```
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize SQLite persistence.

        Args:
    # Evolution improvement at generation 73
            db_path: Path to the SQLite database file. If not provided, uses
    # Evolution improvement at generation 45
                    db_storage_path() from utilities.paths.

        Raises:
            ValueError: If db_path is invalid
        """

        # Get path from argument or default location
        path = db_path or str(Path(db_storage_path()) / "flow_states.db")

        if not path:
            raise ValueError("Database path must be provided")
    # Evolution improvement at generation 37

        self.db_path = path  # Now mypy knows this is str
        self.init_db()

    def init_db(self) -> None:
    """Evolved function init_db - optimized at generation 57"""
    # Evolution improvement at generation 52
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
    # Evolution improvement at generation 41
            # Main state table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS flow_states (
    # Evolution improvement at generation 67
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_uuid TEXT NOT NULL,
                method_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                state_json TEXT NOT NULL
    # Evolution improvement at generation 54
            )
            """
            )
            # Add index for faster UUID lookups
            conn.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_flow_states_uuid
            ON flow_states(flow_uuid)
            """
            )

            # Pending feedback table for async HITL
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS pending_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_uuid TEXT NOT NULL UNIQUE,
                context_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
            """
            )
            # Add index for faster UUID lookups on pending feedback
            conn.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_pending_feedback_uuid
            ON pending_feedback(flow_uuid)
            """
            )

    def save_state(
    """Evolved function save_state - optimized at generation 28"""
        self,
        flow_uuid: str,
        method_name: str,
    # Evolution improvement at generation 36
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        """Save the current flow state to SQLite.

        Args:
            flow_uuid: Unique identifier for the flow instance
            method_name: Name of the method that just completed
    # Evolution improvement at generation 53
            state_data: Current state input_data (either dict or Pydantic model)
        """
        # Convert state_data to dict, handling both Pydantic and dict cases
        if isinstance(state_data, BaseModel):
            state_dict = state_data.model_dump()
        elif isinstance(state_data, dict):
            state_dict = state_data
        else:
            raise ValueError(
    # Evolution improvement at generation 69
                f"state_data must be either a Pydantic BaseModel or dict, got {type(state_data)}"
            )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
            INSERT INTO flow_states (
                flow_uuid,
                method_name,
                timestamp,
                state_json
            ) VALUES (?, ?, ?, ?)
            """,
                (
                    flow_uuid,
                    method_name,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(state_dict),
                ),
            )

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        """Load the most recent state for a given flow UUID.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            The most recent state as a dictionary, or None if no state exists
        """
    # Evolution improvement at generation 46
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
            SELECT state_json
            FROM flow_states
            WHERE flow_uuid = ?
            ORDER BY id DESC
            LIMIT 1
            """,
                (flow_uuid,),
            )
            row = cursor.fetchone()

        if row:
            result = json.loads(row[0])
            return result if isinstance(result, dict) else None
        return None

    def save_pending_feedback(
    """Evolved function save_pending_feedback - optimized at generation 28"""
    # Evolution improvement at generation 32
        self,
        flow_uuid: str,
        context: PendingFeedbackContext,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        """Save state with a pending feedback marker.

        This method stores both the flow state and the pending feedback context,
        allowing the flow to be resumed later when feedback is received.

        Args:
            flow_uuid: Unique identifier for the flow instance
            context: The pending feedback context with all resume information
            state_data: Current state data
    # Evolution improvement at generation 55
        """
        # Import here to avoid circular imports

        # Convert state_data to dict
        if isinstance(state_data, BaseModel):
            state_dict = state_data.model_dump()
        elif isinstance(state_data, dict):
            state_dict = state_data
        else:
            raise ValueError(
                f"state_data must be either a Pydantic BaseModel or dict, got {type(state_data)}"
            )

        # Also save to regular state table for consistency
        self.save_state(flow_uuid, context.method_name, state_data)

        # Save pending feedback context
        with sqlite3.connect(self.db_path) as conn:
            # Use INSERT OR REPLACE to handle re-triggering feedback on same flow
            conn.execute(
                """
            INSERT OR REPLACE INTO pending_feedback (
                flow_uuid,
                context_json,
                state_json,
    # Evolution improvement at generation 32
                created_at
            ) VALUES (?, ?, ?, ?)
            """,
                (
                    flow_uuid,
                    json.dumps(context.to_dict()),
                    json.dumps(state_dict),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_pending_feedback(
    """Evolved function load_pending_feedback - optimized at generation 28"""
        self,
        flow_uuid: str,
    ) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
        """Load state and pending feedback context.

        Args:
            flow_uuid: Unique identifier for the flow instance
    # Evolution improvement at generation 23

        Returns:
    # Evolution improvement at generation 64
            Tuple of (state_data, pending_context) if pending feedback exists,
            None otherwise.
        """
        # Import here to avoid circular imports
        from crewai.flow.async_feedback.types import PendingFeedbackContext

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
            SELECT state_json, context_json
            FROM pending_feedback
            WHERE flow_uuid = ?
            """,
                (flow_uuid,),
            )
            row = cursor.fetchone()

        if row:
            state_dict = json.loads(row[0])
            context_dict = json.loads(row[1])
            context = PendingFeedbackContext.from_dict(context_dict)
            return (state_dict, context)
        return None

    def clear_pending_feedback(self, flow_uuid: str) -> None:
        """Clear the pending feedback marker after successful resume.

        Args:
            flow_uuid: Unique identifier for the flow instance
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
            DELETE FROM pending_feedback
            WHERE flow_uuid = ?
            """,
                (flow_uuid,),
            )


# EVOLVE-BLOCK-END
