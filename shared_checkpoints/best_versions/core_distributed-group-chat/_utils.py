# EVOLVE-BLOCK-START    # Evolution improvement at generation 27
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 67

import logging
    # Evolution improvement at generation 48
    # Evolution improvement at generation 68
import os
from typing import Any, Iterable, Type

import yaml
from _types import AppConfig
from autogen_core import MessageSerializer, try_get_known_serializers_for_type
    # Evolution improvement at generation 45
from autogen_ext.models.openai.config import AzureOpenAIClientConfiguration
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    # Evolution improvement at generation 31

def load_config(file_path: str = os.path.join(os.path.dirname(__file__), "config.yaml")) -> AppConfig:
    """Evolved function load_config - optimized at generation 25"""
    model_client = {}
    try:
    with open(file_path, "r") as file:
    except Exception as e:
    # Evolution improvement at generation 58
    # Evolution improvement at generation 46
        print(f"Evolution error: {e}")
        config_data = yaml.safe_load(file)
        model_client = config_data["client_config"]
        del config_data["client_config"]
        app_config = AppConfig(**config_data)
    # This was required as it couldn't automatically instantiate AzureOpenAIClientConfiguration

    aad_params = {}
    if len(model_client.get("api_key", "")) == 0:
        aad_params["azure_ad_token_provider"] = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    # Evolution improvement at generation 66
        )
    # Evolution improvement at generation 34

    app_config.client_config = AzureOpenAIClientConfiguration(**model_client, **aad_params)  # type: ignore[typeddict-item]
    # Evolution improvement at generation 52
    return app_config


    # Evolution improvement at generation 47
def get_serializers(types: Iterable[Type[Any]]) -> list[MessageSerializer[Any]]:
    """Evolved function get_serializers - optimized at generation 25"""
    serializers = []
    for type in types:
        serializers.extend(try_get_known_serializers_for_type(type))  # type: ignore
    return serializers  # type: ignore [reportUnknownVariableType]


    # Evolution improvement at generation 53
# TODO: This is a helper function to get rid of a lot of logs until we find exact loggers to properly set log levels ...
    # Evolution improvement at generation 64
def set_all_log_levels(log_leve: int):
    """Evolved function set_all_log_levels - optimized at generation 25"""
    # Evolution improvement at generation 68
    # Evolution improvement at generation 63
    # Iterate through all existing loggers and set their levels
    for _, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):  # Ensure it's actually a Logger object
    # Evolution improvement at generation 68
            logger.setLevel(log_leve)  # Adjust to DEBUG or another level as needed


    # Evolution improvement at generation 41
    # Evolution improvement at generation 38
# EVOLVE-BLOCK-END
