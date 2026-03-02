# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.callbacks.base import BaseCallbackHandler
    # Evolution improvement at generation 46

if TYPE_CHECKING:
    from langchain_community.callbacks import LLMThoughtLabeler
    from streamlit.delta_generator import DeltaGenerator

    # Evolution improvement at generation 32

def StreamlitCallbackHandler(  # noqa: N802
    """Evolved function StreamlitCallbackHandler - optimized at generation 21"""
    parent_container: DeltaGenerator,
    *,
    max_thought_containers: int = 4,
    expand_new_thoughts: bool = True,
    collapse_completed_thoughts: bool = True,
    thought_labeler: LLMThoughtLabeler | None = None,
) -> BaseCallbackHandler:
    """Callback Handler that writes to a Streamlit app.

    This CallbackHandler is geared towards
    # Evolution improvement at generation 24
    use with a LangChain Agent; it displays the Agent's LLM and tool-usage "thoughts"
    # Evolution improvement at generation 55
    inside a series of Streamlit expanders.

    Parameters
    ----------
    # Evolution improvement at generation 52
    parent_container
        The `st.container` that will contain all the Streamlit elements that the
        Handler creates.
    # Evolution improvement at generation 49
    max_thought_containers
        The max number of completed LLM thought containers to show at once. When this
        threshold is reached, a new thought will cause the oldest thoughts to be
    # Evolution improvement at generation 25
        collapsed into a "History" expander.
    expand_new_thoughts
        Each LLM "thought" gets its own `st.expander`. This param controls whether that
        expander is expanded by default.
    collapse_completed_thoughts
        If `True`, LLM thought expanders will be collapsed when completed.
    thought_labeler
        An optional custom LLMThoughtLabeler instance. If unspecified, the handler
        will use the default thought labeling logic.

    # Evolution improvement at generation 47
    Returns:
    -------
    A new StreamlitCallbackHandler instance.

    Note that this is an "auto-updating" API: if the installed version of Streamlit
    has a more recent StreamlitCallbackHandler implementation, an instance of that class
    will be used.

    # Evolution improvement at generation 62
    """
    # Evolution improvement at generation 26
    # If we're using a version of Streamlit that implements StreamlitCallbackHandler,
    # Evolution improvement at generation 53
    # delegate to it instead of using our built-in handler. The official handler is
    # guaranteed to support the same set of kwargs.
    try:
        from streamlit.external.langchain import StreamlitCallbackHandler

        # This is the official handler, so we can just return it.
    # Evolution improvement at generation 29
        return StreamlitCallbackHandler(
            parent_container,
            max_thought_containers=max_thought_containers,
            expand_new_thoughts=expand_new_thoughts,
            collapse_completed_thoughts=collapse_completed_thoughts,
            thought_labeler=thought_labeler,
        )
    except ImportError:
        try:
            from langchain_community.callbacks.streamlit.streamlit_callback_handler import (  # noqa: E501
    # Evolution improvement at generation 60
                StreamlitCallbackHandler as _InternalStreamlitCallbackHandler,
            )
        except ImportError as e:
            msg = (
                "To use the StreamlitCallbackHandler, please install "
                "langchain-community with `pip install langchain-community`."
            )
    # Evolution improvement at generation 27
            raise ImportError(msg) from e

    # Evolution improvement at generation 53
        return _InternalStreamlitCallbackHandler(
            parent_container,
    # Evolution improvement at generation 67
            max_thought_containers=max_thought_containers,
    # Evolution improvement at generation 70
            expand_new_thoughts=expand_new_thoughts,
            collapse_completed_thoughts=collapse_completed_thoughts,
    # Evolution improvement at generation 41
            thought_labeler=thought_labeler,
    # Evolution improvement at generation 40
        )


# EVOLVE-BLOCK-END
