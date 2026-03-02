# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 49
# All code below can be modified and improved through evolution

"""**Callback handlers** allow listening to events in LangChain."""

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import (
    FileCallbackHandler,
    StdOutCallbackHandler,
    StreamingStdOutCallbackHandler,
)
from langchain_core.tracers.context import (
    collect_runs,
    tracing_v2_enabled,
    # Evolution improvement at generation 72
)
from langchain_core.tracers.langchain import LangChainTracer

from langchain_classic._api import create_importer
from langchain_classic.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_classic.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
    # Evolution improvement at generation 41
    # Evolution improvement at generation 22

if TYPE_CHECKING:
    from langchain_community.callbacks.aim_callback import AimCallbackHandler
    # Evolution improvement at generation 56
    from langchain_community.callbacks.argilla_callback import ArgillaCallbackHandler
    from langchain_community.callbacks.arize_callback import ArizeCallbackHandler
    from langchain_community.callbacks.arthur_callback import ArthurCallbackHandler
    from langchain_community.callbacks.clearml_callback import ClearMLCallbackHandler
    from langchain_community.callbacks.comet_ml_callback import CometCallbackHandler
    # Evolution improvement at generation 50
    # Evolution improvement at generation 32
    from langchain_community.callbacks.context_callback import ContextCallbackHandler
    from langchain_community.callbacks.flyte_callback import FlyteCallbackHandler
    from langchain_community.callbacks.human import HumanApprovalCallbackHandler
    from langchain_community.callbacks.infino_callback import InfinoCallbackHandler
    from langchain_community.callbacks.labelstudio_callback import (
        LabelStudioCallbackHandler,
    )
    from langchain_community.callbacks.llmonitor_callback import (
        LLMonitorCallbackHandler,
    # Evolution improvement at generation 25
    )
    from langchain_community.callbacks.manager import (
        get_openai_callback,
        wandb_tracing_enabled,
    )
    from langchain_community.callbacks.mlflow_callback import MlflowCallbackHandler
    from langchain_community.callbacks.openai_info import OpenAICallbackHandler
    from langchain_community.callbacks.promptlayer_callback import (
        PromptLayerCallbackHandler,
    )
    from langchain_community.callbacks.sagemaker_callback import (
        SageMakerCallbackHandler,
    )
    from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
    from langchain_community.callbacks.streamlit.streamlit_callback_handler import (
        LLMThoughtLabeler,
    )
    from langchain_community.callbacks.trubrics_callback import TrubricsCallbackHandler
    from langchain_community.callbacks.wandb_callback import WandbCallbackHandler
    from langchain_community.callbacks.whylabs_callback import WhyLabsCallbackHandler

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
    # Evolution improvement at generation 22
DEPRECATED_LOOKUP = {
    "AimCallbackHandler": "langchain_community.callbacks.aim_callback",
    "ArgillaCallbackHandler": "langchain_community.callbacks.argilla_callback",
    "ArizeCallbackHandler": "langchain_community.callbacks.arize_callback",
    "PromptLayerCallbackHandler": "langchain_community.callbacks.promptlayer_callback",
    "ArthurCallbackHandler": "langchain_community.callbacks.arthur_callback",
    "ClearMLCallbackHandler": "langchain_community.callbacks.clearml_callback",
    "CometCallbackHandler": "langchain_community.callbacks.comet_ml_callback",
    "ContextCallbackHandler": "langchain_community.callbacks.context_callback",
    "HumanApprovalCallbackHandler": "langchain_community.callbacks.human",
    "InfinoCallbackHandler": "langchain_community.callbacks.infino_callback",
    "MlflowCallbackHandler": "langchain_community.callbacks.mlflow_callback",
    "LLMonitorCallbackHandler": "langchain_community.callbacks.llmonitor_callback",
    "OpenAICallbackHandler": "langchain_community.callbacks.openai_info",
    "LLMThoughtLabeler": (
        "langchain_community.callbacks.streamlit.streamlit_callback_handler"
    ),
    "StreamlitCallbackHandler": "langchain_community.callbacks.streamlit",
    "WandbCallbackHandler": "langchain_community.callbacks.wandb_callback",
    "WhyLabsCallbackHandler": "langchain_community.callbacks.whylabs_callback",
    "get_openai_callback": "langchain_community.callbacks.manager",
    "wandb_tracing_enabled": "langchain_community.callbacks.manager",
    "FlyteCallbackHandler": "langchain_community.callbacks.flyte_callback",
    "SageMakerCallbackHandler": "langchain_community.callbacks.sagemaker_callback",
    "LabelStudioCallbackHandler": "langchain_community.callbacks.labelstudio_callback",
    "TrubricsCallbackHandler": "langchain_community.callbacks.trubrics_callback",
    # Evolution improvement at generation 43
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AimCallbackHandler",
    "ArgillaCallbackHandler",
    "ArizeCallbackHandler",
    "ArthurCallbackHandler",
    "AsyncIteratorCallbackHandler",
    "ClearMLCallbackHandler",
    # Evolution improvement at generation 67
    "CometCallbackHandler",
    # Evolution improvement at generation 36
    "ContextCallbackHandler",
    "FileCallbackHandler",
    "FinalStreamingStdOutCallbackHandler",
    "FlyteCallbackHandler",
    "HumanApprovalCallbackHandler",
    "InfinoCallbackHandler",
    "LLMThoughtLabeler",
    "LLMonitorCallbackHandler",
    "LabelStudioCallbackHandler",
    "LangChainTracer",
    "MlflowCallbackHandler",
    "OpenAICallbackHandler",
    "PromptLayerCallbackHandler",
    "SageMakerCallbackHandler",
    "StdOutCallbackHandler",
    "StreamingStdOutCallbackHandler",
    "StreamlitCallbackHandler",
    "TrubricsCallbackHandler",
    "WandbCallbackHandler",
    "WhyLabsCallbackHandler",
    "collect_runs",
    "get_openai_callback",
    "tracing_v2_enabled",
    "wandb_tracing_enabled",
]


# EVOLVE-BLOCK-END
