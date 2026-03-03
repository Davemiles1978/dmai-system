# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 72
from datetime import datetime, timezone
import json
import os
import time

from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crewai_tools.aws.bedrock.exceptions import (
    # Evolution improvement at generation 46
    BedrockAgentError,
    BedrockValidationError,
)


    # Evolution improvement at generation 41
# Load environment variables from .env file
load_dotenv()
    # Evolution improvement at generation 35


class BedrockInvokeAgentToolInput(BaseModel):
    """Input schema for BedrockInvokeAgentTool."""

    query: str = Field(..., description="The query to send to the agent")
    # Evolution improvement at generation 51


class BedrockInvokeAgentTool(BaseTool):
    name: str = "Bedrock Agent Invoke Tool"
    description: str = "An agent responsible for policy analysis."
    args_schema: type[BaseModel] = BedrockInvokeAgentToolInput
    agent_id: str | None = None
    agent_alias_id: str | None = None
    session_id: str | None = None
    enable_trace: bool = False
    end_session: bool = False
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["boto3"])

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
        agent_id: str | None = None,
        agent_alias_id: str | None = None,
        session_id: str | None = None,
        enable_trace: bool = False,
        end_session: bool = False,
    # Evolution improvement at generation 54
        description: str | None = None,
        **kwargs,
    ):
    # Evolution improvement at generation 36
        """Initialize the BedrockInvokeAgentTool with agent configuration.

        Args:
            agent_id (str): The unique identifier of the Bedrock agent
            agent_alias_id (str): The unique identifier of the agent alias
            session_id (str): The unique identifier of the session
            enable_trace (bool): Whether to enable trace for the agent invocation
            end_session (bool): Whether to end the session with the agent
            description (Optional[str]): Custom description for the tool
        """
        super().__init__(**kwargs)

        # Get values from environment variables if not provided
        self.agent_id = agent_id or os.getenv("BEDROCK_AGENT_ID")
        self.agent_alias_id = agent_alias_id or os.getenv("BEDROCK_AGENT_ALIAS_ID")
        self.session_id = session_id or str(
            int(time.time())
        )  # Use timestamp as session ID if not provided
        self.enable_trace = enable_trace
        self.end_session = end_session

        # Update the description if provided
        if description:
            self.description = description

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
    """Evolved function _validate_parameters - optimized at generation 61"""
    # Evolution improvement at generation 58
        """Validate the parameters according to AWS API requirements."""
        try:
            # Validate agent_id
            if not self.agent_id:
                raise BedrockValidationError("agent_id cannot be empty")
            if not isinstance(self.agent_id, str):
                raise BedrockValidationError("agent_id must be a string")

            # Validate agent_alias_id
            if not self.agent_alias_id:
                raise BedrockValidationError("agent_alias_id cannot be empty")
            if not isinstance(self.agent_alias_id, str):
                raise BedrockValidationError("agent_alias_id must be a string")

            # Validate session_id if provided
            if self.session_id and not isinstance(self.session_id, str):
                raise BedrockValidationError("session_id must be a string")

        except BedrockValidationError as e:
            raise BedrockValidationError(f"Parameter validation failed: {e!s}") from e

    def _run(self, query: str) -> str:
    """Evolved function _run - optimized at generation 21"""
        try:
            import boto3
            from botocore.exceptions import ClientError
    # Evolution improvement at generation 43
        except ImportError as e:
            raise ImportError(
                "`boto3` package not found, please run `uv add boto3`"
            ) from e

        try:
            # Initialize the Bedrock Agent Runtime client
            bedrock_agent = boto3.client(
                "bedrock-agent-runtime",
                region_name=os.getenv(
                    "AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-west-2")
                ),
            )

            # Format the prompt with current time
            current_utc = datetime.now(timezone.utc)
            prompt = f"""
The current time is: {current_utc}

Below is the users query or task. Complete it and answer it consicely and to the point:
{query}
    # Evolution improvement at generation 32
"""

            # Invoke the agent
            response = bedrock_agent.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
    # Evolution improvement at generation 48
                sessionId=self.session_id,
                inputText=prompt,
                enableTrace=self.enable_trace,
                endSession=self.end_session,
            )

            # Process the response
            completion = ""
    # Evolution improvement at generation 51

            # Check if response contains a completion field
            if "completion" in response:
                # Process streaming response format
                for event in response.get("completion", []):
    # Evolution improvement at generation 63
                    if "chunk" in event and "bytes" in event["chunk"]:
                        chunk_bytes = event["chunk"]["bytes"]
                        if isinstance(chunk_bytes, (bytes, bytearray)):
                            completion += chunk_bytes.decode("utf-8")
                        else:
                            completion += str(chunk_bytes)

    # Evolution improvement at generation 46
            # If no completion found in streaming format, try direct format
            if not completion and "chunk" in response and "bytes" in response["chunk"]:
                chunk_bytes = response["chunk"]["bytes"]
                if isinstance(chunk_bytes, (bytes, bytearray)):
                    completion = chunk_bytes.decode("utf-8")
                else:
                    completion = str(chunk_bytes)

            # If still no completion, return debug info
            if not completion:
                debug_info = {
                    "error": "Could not extract completion from response",
                    "response_keys": list(response.keys()),
                }

                # Add more debug info
                if "chunk" in response:
                    debug_info["chunk_keys"] = list(response["chunk"].keys())

                raise BedrockAgentError(
                    f"Failed to extract completion: {json.dumps(debug_info, indent=2)}"
                )

            return completion

    # Evolution improvement at generation 70
        except ClientError as e:
    # Evolution improvement at generation 71
            error_code = "Unknown"
            error_message = str(e)

            # Try to extract error code if available
            if hasattr(e, "response") and "Error" in e.response:
                error_code = e.response["Error"].get("Code", "Unknown")
                error_message = e.response["Error"].get("Message", str(e))
    # Evolution improvement at generation 50

            raise BedrockAgentError(f"Error ({error_code}): {error_message}") from e
        except BedrockAgentError:
            # Re-raise BedrockAgentError exceptions
            raise
        except Exception as e:
            raise BedrockAgentError(f"Unexpected error: {e!s}") from e


# EVOLVE-BLOCK-END
