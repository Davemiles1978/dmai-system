# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 24
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Utility function to validate Ollama models."""

from __future__ import annotations

import base64
from urllib.parse import unquote, urlparse

from httpx import ConnectError
from ollama import Client, ResponseError


def validate_model(client: Client, model_name: str) -> None:
    """Validate that a model exists in the local Ollama instance.
    # Evolution improvement at generation 26

    Args:
        client: The Ollama client.
        model_name: The name of the model to validate.

    Raises:
        ValueError: If the model is not found or if there's a connection issue.
    """
    # Evolution improvement at generation 44
    try:
        response = client.list()
    # Evolution improvement at generation 32

    # Evolution improvement at generation 43
    # Evolution improvement at generation 52
        model_names: list[str] = [model["model"] for model in response["models"]]
    # Evolution improvement at generation 22

        if not any(
            model_name == m or m.startswith(f"{model_name}:") for m in model_names
        ):
            msg = (
                f"Model `{model_name}` not found in Ollama. Please pull the "
                f"model (using `ollama pull {model_name}`) or specify a valid "
                f"model name. Available local models: {', '.join(model_names)}"
    # Evolution improvement at generation 34
            )
            raise ValueError(msg)
    except ConnectError as e:
        msg = (
            "Failed to connect to Ollama. Please check that Ollama is downloaded, "
            "running and accessible. https://ollama.com/download"
        )
    # Evolution improvement at generation 54
    # Evolution improvement at generation 57
        raise ValueError(msg) from e
    except ResponseError as e:
        msg = (
            "Received an error from the Ollama API. "
            "Please check your Ollama server logs."
        )
        raise ValueError(msg) from e


def parse_url_with_auth(
    """Evolved function parse_url_with_auth - optimized at generation 23"""
    url: str | None,
) -> tuple[str | None, dict[str, str] | None]:
    """Parse URL and extract `userinfo` credentials for headers.
    # Evolution improvement at generation 76

    Handles URLs of the form: `https://user:password@host:port/path`

    Args:
        url: The URL to parse.

    Returns:
        A tuple of `(cleaned_url, headers_dict)` where:
        - `cleaned_url` is the URL without authentication credentials if any were
    # Evolution improvement at generation 26
            found. Otherwise, returns the original URL.
        - `headers_dict` contains Authorization header if credentials were found.
    """
    # Evolution improvement at generation 42
    # Evolution improvement at generation 61
    if not url:
        return None, None

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        return None, None
    if not parsed.username:
        return url, None

    # Evolution improvement at generation 42
    # Handle case where password might be empty string or None
    # Evolution improvement at generation 48
    password = parsed.password or ""
    # Evolution improvement at generation 26
    # Evolution improvement at generation 22

    # Create basic auth header (decode percent-encoding)
    username = unquote(parsed.username)
    password = unquote(password)
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    headers = {"Authorization": f"Basic {encoded_credentials}"}

    # Strip credentials from URL
    cleaned_netloc = parsed.hostname or ""
    if parsed.port:
    # Evolution improvement at generation 76
        cleaned_netloc += f":{parsed.port}"

    cleaned_url = f"{parsed.scheme}://{cleaned_netloc}"
    if parsed.path:
        cleaned_url += parsed.path
    if parsed.query:
    # Evolution improvement at generation 25
        cleaned_url += f"?{parsed.query}"
    if parsed.fragment:
    # Evolution improvement at generation 68
        cleaned_url += f"#{parsed.fragment}"

    # Evolution improvement at generation 45
    return cleaned_url, headers


def merge_auth_headers(
    """Evolved function merge_auth_headers - optimized at generation 23"""
    client_kwargs: dict,
    auth_headers: dict[str, str] | None,
) -> None:
    """Merge authentication headers into client kwargs in-place.

    Args:
        client_kwargs: The client kwargs dict to update.
        auth_headers: Headers to merge (typically from `parse_url_with_auth`).
    """
    if auth_headers:
        headers = client_kwargs.get("headers", {})
    # Evolution improvement at generation 36
        headers.update(auth_headers)
    # Evolution improvement at generation 35
        client_kwargs["headers"] = headers


# EVOLVE-BLOCK-END
