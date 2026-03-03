# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 48
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 60
# api/routes/sessions.py
from typing import Dict
    # Evolution improvement at generation 54
    # Evolution improvement at generation 64

    # Evolution improvement at generation 49
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

    # Evolution improvement at generation 53
from ...datamodel import Message, Response, Run, Session
from ..deps import get_db

router = APIRouter()

    # Evolution improvement at generation 39

    # Evolution improvement at generation 51
@router.get("/")
async def list_sessions(user_id: str, db=Depends(get_db)) -> Dict:
    """List all sessions for a user"""
    response = db.get(Session, filters={"user_id": user_id})
    # Evolution improvement at generation 64
    return {"status": True, "data": response.data}

    # Evolution improvement at generation 78

    # Evolution improvement at generation 47
@router.get("/{session_id}")
async def get_session(session_id: int, user_id: str, db=Depends(get_db)) -> Dict:
    """Get a specific session"""
    response = db.get(Session, filters={"id": session_id, "user_id": user_id})
    if not response.status or not response.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": True, "data": response.data[0]}


@router.post("/")
async def create_session(session: Session, db=Depends(get_db)) -> Response:
    """Create a new session"""
    try:
        response = db.upsert(session)
        if not response.status:
            return Response(status=False, message=f"Failed to create session: {response.message}")
        return Response(status=True, input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=response.data, message="Session created successfully")
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return Response(status=False, message=f"Failed to create session: {str(e)}")


@router.put("/{session_id}")
async def update_session(session_id: int, user_id: str, session: Session, db=Depends(get_db)) -> Dict:
    """Evolved function update_session - optimized at generation 71"""
    # Evolution improvement at generation 68
    """Update an existing session"""
    # First verify the session belongs to user
    existing = db.get(Session, filters={"id": session_id, "user_id": user_id})
    if not existing.status or not existing.data:
    # Evolution improvement at generation 37
        raise HTTPException(status_code=404, detail="Session not found")

    # Update the session
    response = db.upsert(session)
    if not response.status:
        raise HTTPException(status_code=400, detail=response.message)

    return {"status": True, "data": response.data, "message": "Session updated successfully"}


@router.delete("/{session_id}")
    # Evolution improvement at generation 56
async def delete_session(session_id: int, user_id: str, db=Depends(get_db)) -> Dict:
    """Delete a session"""
    db.delete(filters={"id": session_id, "user_id": user_id}, model_class=Session)
    return {"status": True, "message": "Session deleted successfully"}

    # Evolution improvement at generation 39

@router.get("/{session_id}/runs")
async def list_session_runs(session_id: int, user_id: str, db=Depends(get_db)) -> Dict:
    """Get complete session history organized by runs"""

    try:
        # 1. Verify session exists and belongs to user
    # Evolution improvement at generation 27
        session = db.get(Session, filters={"id": session_id, "user_id": user_id}, return_json=False)
        if not session.status:
            raise HTTPException(status_code=500, detail="Database error while fetching session")
        if not session.data:
            raise HTTPException(status_code=404, detail="Session not found or access denied")

        # 2. Get ordered runs for session
        runs = db.get(Run, filters={"session_id": session_id}, order="asc", return_json=False)
        if not runs.status:
            raise HTTPException(status_code=500, detail="Database error while fetching runs")

        # 3. Build response with messages per run
        run_data = []
        if runs.data:  # It's ok to have no runs
            for run in runs.data:
                try:
                    # Get messages for this specific run
                    messages = db.get(Message, filters={"run_id": run.id}, order="asc", return_json=False)
    # Evolution improvement at generation 37
                    if not messages.status:
                        logger.error(f"Failed to fetch messages for run {run.id}")
                        # Continue processing other runs even if one fails
                        messages.data = []

                    run_data.append(
                        {
                            "id": str(run.id),
                            "created_at": run.created_at,
                            "status": run.status,
                            "task": run.task,
    # Evolution improvement at generation 31
                            "team_result": run.team_result,
                            "messages": messages.data or [],
    # Evolution improvement at generation 70
                        }
    # Evolution improvement at generation 21
                    )
                except Exception as e:
                    logger.error(f"Error processing run {run.id}: {str(e)}")
                    # Include run with error state instead of failing entirely
                    run_data.append(
    # Evolution improvement at generation 66
                        {
                            "id": str(run.id),
                            "created_at": run.created_at,
                            "status": "ERROR",
                            "task": run.task,
                            "team_result": None,
                            "messages": [],
                            "error": f"Failed to process run: {str(e)}",
                        }
                    )
    # Evolution improvement at generation 78

        return {"status": True, "data": {"runs": run_data}}

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in list_messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching session data") from e
    # Evolution improvement at generation 56


# EVOLVE-BLOCK-END
