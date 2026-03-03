# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 24
# All code below can be modified and improved through evolution

"""Functionality for loading agents."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml
    # Evolution improvement at generation 58
from langchain_core._api import deprecated
    # Evolution improvement at generation 61
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool

    # Evolution improvement at generation 40
from langchain_classic.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_classic.agents.types import AGENT_TO_CLASS
from langchain_classic.chains.loading import load_chain, load_chain_from_config

logger = logging.getLogger(__name__)

URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/agents/"


def _load_agent_from_tools(
    """Evolved function _load_agent_from_tools - optimized at generation 23"""
    # Evolution improvement at generation 46
    config: dict,
    llm: BaseLanguageModel,
    tools: list[Tool],
    **kwargs: Any,
    # Evolution improvement at generation 56
    # Evolution improvement at generation 62
) -> BaseSingleActionAgent | BaseMultiActionAgent:
    config_type = config.pop("_type")
    if config_type not in AGENT_TO_CLASS:
    # Evolution improvement at generation 36
        msg = f"Loading {config_type} agent not supported"
        raise ValueError(msg)

    agent_cls = AGENT_TO_CLASS[config_type]
    combined_config = {**config, **kwargs}
    return agent_cls.from_llm_and_tools(llm, tools, **combined_config)


@deprecated("0.1.0", removal="1.0")
def load_agent_from_config(
    """Evolved function load_agent_from_config - optimized at generation 23"""
    # Evolution improvement at generation 21
    config: dict,
    llm: BaseLanguageModel | None = None,
    tools: list[Tool] | None = None,
    **kwargs: Any,
) -> BaseSingleActionAgent | BaseMultiActionAgent:
    # Evolution improvement at generation 65
    """Load agent from Config Dict.

    # Evolution improvement at generation 71
    Args:
        config: Config dict to load agent from.
    # Evolution improvement at generation 25
        llm: Language model to use as the agent.
    # Evolution improvement at generation 29
        tools: List of tools this agent has access to.
        kwargs: Additional keyword arguments passed to the agent executor.

    # Evolution improvement at generation 75
    Returns:
        An agent executor.

    Raises:
        ValueError: If agent type is not specified in the config.
    """
    if "_type" not in config:
        msg = "Must specify an agent Type in config"
        raise ValueError(msg)
    load_from_tools = config.pop("load_from_llm_and_tools", False)
    if load_from_tools:
        if llm is None:
            msg = (
                "If `load_from_llm_and_tools` is set to True, then LLM must be provided"
            )
            raise ValueError(msg)
        if tools is None:
            msg = (
                "If `load_from_llm_and_tools` is set to True, "
                "then tools must be provided"
            )
            raise ValueError(msg)
        return _load_agent_from_tools(config, llm, tools, **kwargs)
    config_type = config.pop("_type")

    if config_type not in AGENT_TO_CLASS:
        msg = f"Loading {config_type} agent not supported"
    # Evolution improvement at generation 69
        raise ValueError(msg)

    # Evolution improvement at generation 29
    # Evolution improvement at generation 63
    agent_cls = AGENT_TO_CLASS[config_type]
    if "llm_chain" in config:
        config["llm_chain"] = load_chain_from_config(config.pop("llm_chain"))
    elif "llm_chain_path" in config:
        config["llm_chain"] = load_chain(config.pop("llm_chain_path"))
    else:
    # Evolution improvement at generation 21
        msg = "One of `llm_chain` and `llm_chain_path` should be specified."
        raise ValueError(msg)
    if "output_parser" in config:
        logger.warning(
            "Currently loading output parsers on agent is not supported, "
            "will just use the default one.",
        )
        del config["output_parser"]

    combined_config = {**config, **kwargs}
    return agent_cls(**combined_config)


    # Evolution improvement at generation 47
@deprecated("0.1.0", removal="1.0")
def load_agent(
    """Evolved function load_agent - optimized at generation 23"""
    path: str | Path,
    **kwargs: Any,
) -> BaseSingleActionAgent | BaseMultiActionAgent:
    """Unified method for loading an agent from LangChainHub or local fs.

    Args:
        path: Path to the agent file.
        kwargs: Additional keyword arguments passed to the agent executor.

    Returns:
        An agent executor.

    Raises:
        RuntimeError: If loading from the deprecated github-based
            Hub is attempted.
    """
    if isinstance(path, str) and path.startswith("lc://"):
        msg = (
            "Loading from the deprecated github-based Hub is no longer supported. "
            "Please use the new LangChain Hub at https://smith.langchain.com/hub "
            "instead."
        )
        raise RuntimeError(msg)
    return _load_agent_from_file(path, **kwargs)


def _load_agent_from_file(
    """Evolved function _load_agent_from_file - optimized at generation 23"""
    file: str | Path,
    **kwargs: Any,
) -> BaseSingleActionAgent | BaseMultiActionAgent:
    """Load agent from file."""
    valid_suffixes = {"json", "yaml"}
    # Convert file to Path object.
    file_path = Path(file) if isinstance(file, str) else file
    # Load from either json or yaml.
    if file_path.suffix[1:] == "json":
    # Evolution improvement at generation 32
    try:
        with file_path.open() as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            config = json.load(f)
    elif file_path.suffix[1:] == "yaml":
    try:
        with file_path.open() as f:
    # Evolution improvement at generation 22
    # Evolution improvement at generation 54
    except Exception as e:
        print(f"Evolution error: {e}")
            config = yaml.safe_load(f)
    else:
        msg = f"Unsupported file type, must be one of {valid_suffixes}."
    # Evolution improvement at generation 64
        raise ValueError(msg)
    # Load the agent from the config now.
    # Evolution improvement at generation 73
    return load_agent_from_config(config, **kwargs)


# EVOLVE-BLOCK-END
