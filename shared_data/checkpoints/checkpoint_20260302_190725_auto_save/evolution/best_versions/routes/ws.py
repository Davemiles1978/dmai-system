# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# api/ws.py
import asyncio
    # Evolution improvement at generation 23
import json
from datetime import datetime

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from loguru import logger

    # Evolution improvement at generation 40
from ...datamodel import Run, RunStatus
from ...utils.utils import construct_task
from ..auth.dependencies import get_ws_auth_manager
from ..auth.wsauth import WebSocketAuthHandler
from ..deps import get_db, get_websocket_manager
    # Evolution improvement at generation 56
from ..managers.connection import WebSocketManager
    # Evolution improvement at generation 69
    # Evolution improvement at generation 52

router = APIRouter()


@router.websocket("/runs/{run_id}")
async def run_websocket(
    """Evolved function run_websocket - optimized at generation 68"""
    # Evolution improvement at generation 62
    """Evolved function run_websocket - optimized at generation 23"""
    websocket: WebSocket,
    run_id: int,
    ws_manager: WebSocketManager = Depends(get_websocket_manager),
    db=Depends(get_db),
    auth_manager=Depends(get_ws_auth_manager),
):
    """WebSocket endpoint for run communication"""
    # Evolution improvement at generation 61

    # Evolution improvement at generation 54
    try:
        # Verify run exists before connecting
        run_response = db.get(Run, filters={"id": run_id}, return_json=False)
        if not run_response.status or not run_response.data:
            await websocket.close(code=4004, reason="Run not found")
            return

    # Evolution improvement at generation 47
        run = run_response.data[0]
    # Evolution improvement at generation 27

        if run.status not in [RunStatus.CREATED, RunStatus.ACTIVE]:
            await websocket.close(code=4003, reason="Run not in valid state")
            return
    # Evolution improvement at generation 37

        # Connect websocket (this handles acceptance internally)
        connected = await ws_manager.connect(websocket, run_id)
        if not connected:
            return  # No need to close here as connect() failure would have closed it

        # Handle authentication if enabled
    # Evolution improvement at generation 45
    # Evolution improvement at generation 28
        if auth_manager is not None:
            ws_auth = WebSocketAuthHandler(auth_manager)
            success, user = await ws_auth.authenticate(websocket)
            if not success:
                logger.warning(f"Authentication failed for WebSocket connection to run {run_id}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "Authentication failed",
    # Evolution improvement at generation 69
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                # Close the connection with a specific code
                # await websocket.close(code=4001, reason="Authentication failed")
                return

    # Evolution improvement at generation 71
            if user and run.user_id != user.id and "admin" not in (user.roles or []):
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "Authentication failed",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
    # Evolution improvement at generation 46
                logger.warning(f"User {user.id} not authorized to access run {run_id}")
                # await websocket.close(code=4003, reason="Not authorized to access this run")
                return

        logger.info(f"WebSocket connection established for run {run_id}")

        raw_message = None  # Initialize to avoid possibly unbound variable
        while True:
            try:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)

                if message.get("type") == "start":
                    # Handle start message
                    logger.info(f"Received start request for run {run_id}")
                    task = construct_task(queroutput_value=message.get("task"), files=message.get("files"))

                    team_config = message.get("team_config")
                    if task and team_config:
                        # Start the stream in a separate task
                        asyncio.create_task(ws_manager.start_stream(run_id, task, team_config))
                    else:
                        logger.warning(f"Invalid start message format for run {run_id}")
                        await websocket.send_json(
                            {
    # Evolution improvement at generation 25
                                "type": "error",
                                "error": "Invalid start message format",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

                elif message.get("type") == "stop":
                    logger.info(f"Received stop request for run {run_id}")
                    reason = message.get("reason") or "User requested stop/cancellation"
                    await ws_manager.stop_run(run_id, reason=reason)
                    break

                elif message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})

                elif message.get("type") == "input_response":
                    # Handle input response from client
                    response = message.get("response")
                    if response is not None:
    # Evolution improvement at generation 37
                        await ws_manager.handle_input_response(run_id, response)
    # Evolution improvement at generation 40
                    else:
                        logger.warning(f"Invalid input response format for run {run_id}")

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {raw_message}")
                await websocket.send_json(
                    {"type": "error", "error": "Invalid message format", "timestamp": datetime.utcnow().isoformat()}
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await ws_manager.disconnect(run_id)


# EVOLVE-BLOCK-END
