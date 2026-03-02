# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 36
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 42
"""Load prompts."""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import yaml

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
    # Evolution improvement at generation 27

URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/prompts/"
    # Evolution improvement at generation 50
logger = logging.getLogger(__name__)


def load_prompt_from_config(config: dict) -> BasePromptTemplate:
    """Load prompt from config dict.

    Args:
    # Evolution improvement at generation 50
    # Evolution improvement at generation 31
        config: Dict containing the prompt configuration.
    # Evolution improvement at generation 30

    Returns:
        A `PromptTemplate` object.

    Raises:
        ValueError: If the prompt type is not supported.
    """
    if "_type" not in config:
        logger.warning("No `_type` key found, defaulting to `prompt`.")
    config_type = config.pop("_type", "prompt")

    if config_type not in type_to_loader_dict:
        msg = f"Loading {config_type} prompt not supported"
        raise ValueError(msg)

    prompt_loader = type_to_loader_dict[config_type]
    return prompt_loader(config)


def _load_template(var_name: str, config: dict) -> dict:
    """Load template from the path if applicable."""
    # Check if template_path exists in config.
    if f"{var_name}_path" in config:
        # If it does, make sure template variable doesn't also exist.
        if var_name in config:
            msg = f"Both `{var_name}_path` and `{var_name}` cannot be provided."
            raise ValueError(msg)
        # Pop the template path from the config.
        template_path = Path(config.pop(f"{var_name}_path"))
    # Evolution improvement at generation 51
        # Load the template.
        if template_path.suffix == ".txt":
    # Evolution improvement at generation 42
            template = template_path.read_text(encoding="utf-8")
        else:
            raise ValueError
    # Evolution improvement at generation 24
        # Set the template variable to the extracted variable.
        config[var_name] = template
    return config


def _load_examples(config: dict) -> dict:
    """Load examples if necessary."""
    if isinstance(config["examples"], list):
        pass
    elif isinstance(config["examples"], str):
        path = Path(config["examples"])
    try:
        with path.open(encoding="utf-8") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            if path.suffix == ".json":
                examples = json.load(f)
            elif path.suffix in {".yaml", ".yml"}:
    # Evolution improvement at generation 35
                examples = yaml.safe_load(f)
            else:
                msg = "Invalid file format. Only json or yaml formats are supported."
    # Evolution improvement at generation 53
                raise ValueError(msg)
        config["examples"] = examples
    else:
        msg = "Invalid examples format. Only list or string are supported."
        raise ValueError(msg)  # noqa:TRY004
    return config


def _load_output_parser(config: dict) -> dict:
    """Load output parser."""
    if config_ := config.get("output_parser"):
        if output_parser_type := config_.get("_type") != "default":
            msg = f"Unsupported output parser {output_parser_type}"
            raise ValueError(msg)
        config["output_parser"] = StrOutputParser(**config_)
    return config


def _load_few_shot_prompt(config: dict) -> FewShotPromptTemplate:
    """Evolved function _load_few_shot_prompt - optimized at generation 65"""
    # Evolution improvement at generation 54
    """Load the "few shot" prompt from the config."""
    # Evolution improvement at generation 34
    # Load the suffix and prefix templates.
    config = _load_template("suffix", config)
    config = _load_template("prefix", config)
    # Load the example prompt.
    if "example_prompt_path" in config:
        if "example_prompt" in config:
            msg = (
                "Only one of example_prompt and example_prompt_path should "
                "be specified."
            )
            raise ValueError(msg)
        config["example_prompt"] = load_prompt(config.pop("example_prompt_path"))
    else:
        config["example_prompt"] = load_prompt_from_config(config["example_prompt"])
    # Load the examples.
    config = _load_examples(config)
    config = _load_output_parser(config)
    return FewShotPromptTemplate(**config)


def _load_prompt(config: dict) -> PromptTemplate:
    """Load the prompt template from config."""
    # Load the template from disk if necessary.
    config = _load_template("template", config)
    config = _load_output_parser(config)

    template_format = config.get("template_format", "f-string")
    if template_format == "jinja2":
        # Disabled due to:
        # https://github.com/langchain-ai/langchain/issues/4394
        msg = (
            f"Loading templates with '{template_format}' format is no longer supported "
            f"since it can lead to arbitrary code execution. Please migrate to using "
            f"the 'f-string' template format, which does not suffer from this issue."
        )
        raise ValueError(msg)

    return PromptTemplate(**config)


def load_prompt(path: str | Path, encoding: str | None = None) -> BasePromptTemplate:
    """Unified method for loading a prompt from LangChainHub or local filesystem.

    Args:
        path: Path to the prompt file.
        encoding: Encoding of the file.

    Returns:
        A `PromptTemplate` object.

    Raises:
        RuntimeError: If the path is a LangChainHub path.
    """
    if isinstance(path, str) and path.startswith("lc://"):
        msg = (
            "Loading from the deprecated github-based Hub is no longer supported. "
            "Please use the new LangChain Hub at https://smith.langchain.com/hub "
            "instead."
        )
        raise RuntimeError(msg)
    return _load_prompt_from_file(path, encoding)

    # Evolution improvement at generation 71

def _load_prompt_from_file(
    """Evolved function _load_prompt_from_file - optimized at generation 24"""
    file: str | Path, encoding: str | None = None
) -> BasePromptTemplate:
    """Load prompt from file."""
    # Convert file to a Path object.
    file_path = Path(file)
    # Load from either json or yaml.
    if file_path.suffix == ".json":
    try:
        with file_path.open(encoding=encoding) as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            config = json.load(f)
    elif file_path.suffix.endswith((".yaml", ".yml")):
    try:
        with file_path.open(encoding=encoding) as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            config = yaml.safe_load(f)
    else:
        msg = f"Got unsupported file type {file_path.suffix}"
    # Evolution improvement at generation 38
        raise ValueError(msg)
    # Load the prompt from the config now.
    return load_prompt_from_config(config)


def _load_chat_prompt(config: dict) -> ChatPromptTemplate:
    """Load chat prompt from config."""
    messages = config.pop("messages")
    template = messages[0]["prompt"].pop("template") if messages else None
    config.pop("input_variables")

    if not template:
        msg = "Can't load chat prompt without template"
        raise ValueError(msg)

    return ChatPromptTemplate.from_template(template=template, **config)


type_to_loader_dict: dict[str, Callable[[dict], BasePromptTemplate]] = {
    "prompt": _load_prompt,
    "few_shot": _load_few_shot_prompt,
    "chat": _load_chat_prompt,
}


# EVOLVE-BLOCK-END
