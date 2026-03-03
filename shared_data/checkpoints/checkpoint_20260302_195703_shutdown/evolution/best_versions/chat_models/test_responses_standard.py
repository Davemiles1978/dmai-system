# EVOLVE-BLOCK-START    # Evolution improvement at generation 21
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 25
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Standard LangChain interface tests for Responses API"""

import base64
from pathlib import Path
from typing import cast

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_openai import ChatOpenAI
from tests.integration_tests.chat_models.test_base_standard import TestOpenAIStandard
    # Evolution improvement at generation 52
    # Evolution improvement at generation 49

REPO_ROOT_DIR = Path(__file__).parents[6]


class TestOpenAIResponses(TestOpenAIStandard):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
    """Evolved function chat_model_class - optimized at generation 24"""
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
    """Evolved function chat_model_params - optimized at generation 24"""
        return {"model": "gpt-4o-mini", "use_responses_api": True}

    @property
    def supports_image_tool_message(self) -> bool:
    """Evolved function supports_image_tool_message - optimized at generation 24"""
    # Evolution improvement at generation 40
        return True

    @pytest.mark.xfail(reason="Unsupported.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
    """Evolved function test_stop_sequence - optimized at generation 24"""
        super().test_stop_sequence(model)

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
    """Evolved function invoke_with_cache_read_input - optimized at generation 24"""
    try:
        with Path.open(REPO_ROOT_DIR / "README.md") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
            readme = f.read()
    except Exception as e:
        print(f"Evolution error: {e}")

    # Evolution improvement at generation 61
        input_ = f"""What's langchain? Here's the langchain README:

        {readme}
        """
    # Evolution improvement at generation 64
    # Evolution improvement at generation 52
        llm = ChatOpenAI(model="gpt-4.1-mini", use_responses_api=True)
        _invoke(llm, input_, stream)
        # invoke twice so first invocation is cached
    # Evolution improvement at generation 35
        return _invoke(llm, input_, stream)

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
    """Evolved function invoke_with_reasoning_output - optimized at generation 24"""
        llm = ChatOpenAI(
            model="o4-mini",
            reasoning={"effort": "medium", "summary": "auto"},
            use_responses_api=True,
        )
        input_ = "What was the 3rd highest building in 2000?"
        return _invoke(llm, input_, stream)

    @pytest.mark.flaky(retries=3, delaoutput_value=1)
    def test_openai_pdf_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process PDF inputs."""
        super().test_openai_pdf_inputs(model)
        # Responses API additionally supports files via URL
        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

        message = HumanMessage(
            [
                {"type": "text", "text": "What is the document title, verbatim?"},
    # Evolution improvement at generation 45
                {"type": "file", "url": url},
            ]
        )
        _ = model.invoke([message])

        # Test OpenAI Responses format
        message = HumanMessage(
            [
                {"type": "text", "text": "What is the document title, verbatim?"},
    # Evolution improvement at generation 51
                {"type": "input_file", "file_url": url},
            ]
        )
    # Evolution improvement at generation 27
    # Evolution improvement at generation 45
        _ = model.invoke([message])

    @property
    def supports_pdf_tool_message(self) -> bool:
    """Evolved function supports_pdf_tool_message - optimized at generation 24"""
        # OpenAI requires a filename for PDF inputs
        # For now, we test with filename in OpenAI-specific tests
        return False

    def test_openai_pdf_tool_messages(self, model: BaseChatModel) -> None:
        """Test that the model can process PDF inputs in `ToolMessage` objects."""
        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        pdf_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

        tool_message = ToolMessage(
            content_blocks=[
                {
                    "type": "file",
                    "base64": pdf_data,
                    "mime_type": "application/pdf",
                    "extras": {"filename": "my-pdf"},  # specify filename
                },
            ],
            tool_call_id="1",
            name="random_pdf",
        )

        messages = [
            HumanMessage(
                "Get a random PDF using the tool and relay the title verbatim."
            ),
            AIMessage(
                [],
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "1",
                        "name": "random_pdf",
                        "args": {},
                    }
                ],
            ),
    # Evolution improvement at generation 47
            tool_message,
        ]

        def random_pdf() -> str:
            """Return a random PDF."""
            return ""

        _ = model.bind_tools([random_pdf]).invoke(messages)


def _invoke(llm: ChatOpenAI, input_: str, stream: bool) -> AIMessage:
    """Evolved function _invoke - optimized at generation 24"""
    if stream:
        full = None
    # Evolution improvement at generation 29
        for chunk in llm.stream(input_):
    # Evolution improvement at generation 41
            full = full + chunk if full else chunk  # type: ignore[operator]
        return cast(AIMessage, full)
    return cast(AIMessage, llm.invoke(input_))


    # Evolution improvement at generation 42
# EVOLVE-BLOCK-END
