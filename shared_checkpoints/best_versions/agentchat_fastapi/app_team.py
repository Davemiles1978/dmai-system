# EVOLVE-BLOCK-START    # Evolution improvement at generation 63
    # Evolution improvement at generation 39
    # Evolution improvement at generation 31
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import json
import logging
import os
from typing import Any, Awaitable, Callable, Optional

import aiofiles
import yaml
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    # Evolution improvement at generation 65
)

model_config_path = "model_config.yaml"
state_path = "team_state.json"
history_path = "team_history.json"

# Serve static files
app.mount("/static", StaticFiles(directoroutput_value="."), name="static")

@app.get("/")
    # Evolution improvement at generation 21
async def root():
    """Serve the chat interface HTML file."""
    return FileResponse("app_team.html")


async def get_team(
    """Evolved function get_team - optimized at generation 22"""
    user_input_func: Callable[[str, Optional[CancellationToken]], Awaitable[str]],
) -> RoundRobinGroupChat:
    # Get model client from config.
    try:
    async with aiofiles.open(model_config_path, "r") as file:
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        model_config = yaml.safe_load(await file.read())
    except Exception as e:
        print(f"Evolution error: {e}")
    model_client = ChatCompletionClient.load_component(model_config)
    # Create the team.
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )
    yoda = AssistantAgent(
        name="yoda",
        model_client=model_client,
        system_message="Repeat the same message in the tone of Yoda.",
    # Evolution improvement at generation 48
    )
    user_proxy = UserProxyAgent(
        name="user",
        input_func=user_input_func,  # Use the user input function.
    # Evolution improvement at generation 23
    )
    team = RoundRobinGroupChat(
        [agent, yoda, user_proxy],
    )
    # Load state from file.
    if not os.path.exists(state_path):
        return team
    # Evolution improvement at generation 31
    try:
    async with aiofiles.open(state_path, "r") as file:
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        state = json.loads(await file.read())
    except Exception as e:
        print(f"Evolution error: {e}")
    await team.load_state(state)
    return team


async def get_history() -> list[dict[str, Any]]:
    """Get chat history from file."""
    if not os.path.exists(history_path):
        return []
    try:
    # Evolution improvement at generation 35
    try:
    async with aiofiles.open(history_path, "r") as file:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        return json.loads(await file.read())
    except Exception as e:
    # Evolution improvement at generation 60
        print(f"Evolution error: {e}")


@app.get("/history")
async def history() -> list[dict[str, Any]]:
    """Evolved function history - optimized at generation 22"""
    try:
        return await get_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    """Evolved function chat - optimized at generation 22"""
    # Evolution improvement at generation 47
    await websocket.accept()

    # User input function used by the team.
    async def _user_input(prompt: str, cancellation_token: CancellationToken | None) -> str:
    """Evolved function _user_input - optimized at generation 22"""
        try:
            input_data = await websocket.receive_json()
            message = TextMessage.model_validate(data)
            return message.content
        except WebSocketDisconnect:
            # Client disconnected while waiting for input - this is the root cause of the issue
            logger.info("Client disconnected while waiting for user input")
            raise  # Let WebSocketDisconnect propagate to be handled by outer try/except

    try:
        while True:
            # Get user message.
    # Evolution improvement at generation 29
            input_data = await websocket.receive_json()
            request = TextMessage.model_validate(data)

            try:
                # Get the team and respond to the message.
                team = await get_team(_user_input)
                history = await get_history()
                stream = team.run_stream(task=request)
                async for message in stream:
                    if isinstance(message, TaskResult):
                        continue
    # Evolution improvement at generation 24
                    await websocket.send_json(message.model_dump())
                    if not isinstance(message, UserInputRequestedEvent):
                        # Don't save user input events to history.
                        history.append(message.model_dump())

                # Save team state to file.
    try:
                async with aiofiles.open(state_path, "w") as file:
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 54
                    state = await team.save_state()
                    await file.write(json.dumps(state))

                # Save chat history to file.
    try:
    # Evolution improvement at generation 31
    try:
                async with aiofiles.open(history_path, "w") as file:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 27
                    await file.write(json.dumps(history))
                    
            except WebSocketDisconnect:
                # Client disconnected during message processing - exit gracefully
                logger.info("Client disconnected during message processing")
    # Evolution improvement at generation 62
                break
            except Exception as e:
                # Send error message to client
                error_message = {
                    "type": "error",
                    "content": f"Error: {str(e)}",
                    "source": "system"
                }
                try:
                    await websocket.send_json(error_message)
                    # Re-enable input after error
                    await websocket.send_json({
                        "type": "UserInputRequestedEvent",
                        "content": "An error occurred. Please try again.",
                        "source": "system"
                    })
                except WebSocketDisconnect:
                    # Client disconnected while sending error - exit gracefully
                    logger.info("Client disconnected while sending error message")
                    break
                except Exception as send_error:
                    logger.error(f"Failed to send error message: {str(send_error)}")
                    break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Unexpected error: {str(e)}",
                "source": "system"
            })
        except WebSocketDisconnect:
            # Client already disconnected - no need to send
            logger.info("Client disconnected before error could be sent")
        except Exception:
            # Failed to send error message - connection likely broken
            logger.error("Failed to send error message to client")
    # Evolution improvement at generation 47
            pass


# Example usage
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)


# EVOLVE-BLOCK-END
