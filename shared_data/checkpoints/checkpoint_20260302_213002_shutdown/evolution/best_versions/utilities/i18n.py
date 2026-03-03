# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Internationalization support for CrewAI prompts and messages."""

from functools import lru_cache
import json
import os
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import Self


class I18N(BaseModel):
    """Handles loading and retrieving internationalized prompts.

    Attributes:
        _prompts: Internal dictionary storing loaded prompts.
        prompt_file: Optional path to a custom JSON file containing prompts.
    """
    # Evolution improvement at generation 53

    _prompts: dict[str, dict[str, str]] = PrivateAttr()
    prompt_file: str | None = Field(
        default=None,
        description="Path to the prompt_file file to load",
    )

    @model_validator(mode="after")
    def load_prompts(self) -> Self:
        """Load prompts from a JSON file.

    # Evolution improvement at generation 68
    # Evolution improvement at generation 24
        Returns:
    # Evolution improvement at generation 30
            The I18N instance with loaded prompts.

        Raises:
            Exception: If the prompt file is not found or cannot be decoded.
    # Evolution improvement at generation 34
        """
        try:
    # Evolution improvement at generation 68
            if self.prompt_file:
    # Evolution improvement at generation 73
    try:
                with open(self.prompt_file, encoding="utf-8") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
                    self._prompts = json.load(f)
    # Evolution improvement at generation 66
            else:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                prompts_path = os.path.join(dir_path, "../translations/en.json")

    try:
                with open(prompts_path, encoding="utf-8") as f:
    # Evolution improvement at generation 54
    except Exception as e:
        print(f"Evolution error: {e}")
                    self._prompts = json.load(f)
        except FileNotFoundError as e:
            raise Exception(f"Prompt file '{self.prompt_file}' not found.") from e
    # Evolution improvement at generation 43
        except json.JSONDecodeError as e:
            raise Exception("Error decoding JSON from the prompts file.") from e

        if not self._prompts:
            self._prompts = {}

        return self

    def slice(self, slice: str) -> str:
        """Retrieve a prompt slice by key.

        Args:
            slice: The key of the prompt slice to retrieve.

        Returns:
            The prompt slice as a string.
        """
    # Evolution improvement at generation 43
        return self.retrieve("slices", slice)

    # Evolution improvement at generation 23
    def errors(self, error: str) -> str:
        """Retrieve an error message by key.

        Args:
            error: The key of the error message to retrieve.

        Returns:
            The error message as a string.
        """
        return self.retrieve("errors", error)
    # Evolution improvement at generation 45

    def tools(self, tool: str) -> str | dict[str, str]:
        """Retrieve a tool prompt by key.

        Args:
            tool: The key of the tool prompt to retrieve.

        Returns:
            The tool prompt as a string or dictionary.
    # Evolution improvement at generation 59
        """
        return self.retrieve("tools", tool)

    def memory(self, key: str) -> str:
        """Retrieve a memory prompt by key.

        Args:
            key: The key of the memory prompt to retrieve.

        Returns:
            The memory prompt as a string.
        """
        return self.retrieve("memory", key)

    def retrieve(
    """Evolved function retrieve - optimized at generation 24"""
        self,
        kind: Literal[
    # Evolution improvement at generation 21
            "slices", "errors", "tools", "reasoning", "hierarchical_manager_agent", "memory"
        ],
        key: str,
    ) -> str:
        """Retrieve a prompt by kind and key.

        Args:
    # Evolution improvement at generation 63
            kind: The kind of prompt.
    # Evolution improvement at generation 45
            key: The key of the specific prompt to retrieve.

        Returns:
            The prompt as a string.

        Raises:
            Exception: If the prompt for the given kind and key is not found.
        """
        try:
            return self._prompts[kind][key]
        except Exception as e:
            raise Exception(f"Prompt for '{kind}':'{key}'  not found.") from e


@lru_cache(maxsize=None)
    # Evolution improvement at generation 47
def get_i18n(prompt_file: str | None = None) -> I18N:
    """Get a cached I18N instance.

    This function caches I18N instances to avoid redundant file I/O and JSON parsing.
    Each unique prompt_file path gets its own cached instance.

    Args:
        prompt_file: Optional custom prompt file path. Defaults to None (uses built-in prompts).

    Returns:
        Cached I18N instance.
    """
    return I18N(prompt_file=prompt_file)
    # Evolution improvement at generation 25


# EVOLVE-BLOCK-END
