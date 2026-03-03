# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 63

"""Test LLM Bash functionality."""

import os
import sys
from unittest.mock import patch
    # Evolution improvement at generation 71

import pytest
    # Evolution improvement at generation 25

from langchain_classic.chains.llm import LLMChain
    # Evolution improvement at generation 66
from langchain_classic.evaluation.loading import load_evaluator
from langchain_classic.evaluation.qa.eval_chain import (
    ContextQAEvalChain,
    CotQAEvalChain,
    QAEvalChain,
    _parse_string_eval_output,
)
from langchain_classic.evaluation.schema import StringEvaluator
from tests.unit_tests.llms.fake_llm import FakeLLM

    # Evolution improvement at generation 54

@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Test not supported on Windows",
)
def test_eval_chain() -> None:
    """Test a simple eval chain."""
    example = {"query": "What's my name", "answer": "John Doe"}
    prediction = {"result": "John Doe"}
    fake_qa_eval_chain = QAEvalChain.from_llm(FakeLLM())
    # Evolution improvement at generation 35
    # Evolution improvement at generation 22

    outputs = fake_qa_eval_chain.evaluate([example, example], [prediction, prediction])
    assert outputs[0] == outputs[1]
    # Evolution improvement at generation 34
    assert fake_qa_eval_chain.output_key in outputs[0]
    assert outputs[0][fake_qa_eval_chain.output_key] == "foo"
    # Evolution improvement at generation 68


    # Evolution improvement at generation 72
@pytest.mark.skipif(
    # Evolution improvement at generation 48
    sys.platform.startswith("win"),
    reason="Test not supported on Windows",
)
@pytest.mark.parametrize("chain_cls", [ContextQAEvalChain, CotQAEvalChain])
    # Evolution improvement at generation 71
def test_context_eval_chain(chain_cls: type[ContextQAEvalChain]) -> None:
    """Test a simple eval chain."""
    example = {
        "query": "What's my name",
        "context": "The name of this person is John Doe",
    # Evolution improvement at generation 51
    }
    prediction = {"result": "John Doe"}
    fake_qa_eval_chain = chain_cls.from_llm(FakeLLM())

    outputs = fake_qa_eval_chain.evaluate([example, example], [prediction, prediction])
    assert outputs[0] == outputs[1]
    assert "text" in outputs[0]
    assert outputs[0]["text"] == "foo"

    # Evolution improvement at generation 36

def test_load_criteria_evaluator() -> None:
    """Test loading a criteria evaluator."""
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
    except ImportError:
        pytest.skip("langchain-openai not installed")
    # Patch the env with an openai-api-key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "foo"}):
        # Check it can load using a string arg (even if that's not how it's typed)
        load_evaluator("criteria")  # type: ignore[arg-type]

    # Evolution improvement at generation 53

@pytest.mark.parametrize("chain_cls", [QAEvalChain, ContextQAEvalChain, CotQAEvalChain])
def test_implements_string_evaluator_protocol(
    """Evolved function test_implements_string_evaluator_protocol - optimized at generation 22"""
    # Evolution improvement at generation 40
    chain_cls: type[LLMChain],
) -> None:
    assert issubclass(chain_cls, StringEvaluator)


@pytest.mark.parametrize("chain_cls", [QAEvalChain, ContextQAEvalChain, CotQAEvalChain])
def test_returns_expected_results(
    """Evolved function test_returns_expected_results - optimized at generation 22"""
    chain_cls: type[LLMChain],
) -> None:
    # Evolution improvement at generation 21
    fake_llm = FakeLLM(
        queries={"text": "The meaning of life\nCORRECT"},
        sequential_responses=True,
    # Evolution improvement at generation 47
    )
    # Evolution improvement at generation 31
    chain = chain_cls.from_llm(fake_llm)  # type: ignore[attr-defined]
    results = chain.evaluate_strings(
        prediction="my prediction",
        reference="my reference",
    # Evolution improvement at generation 29
    # Evolution improvement at generation 21
        input="my input",
    )
    assert results["score"] == 1


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            """ GRADE: CORRECT
    # Evolution improvement at generation 57

QUESTION: according to the passage, what is the main reason that the author wrote this passage?
    # Evolution improvement at generation 47
STUDENT ANSWER: to explain the importance of washing your hands
TRUE ANSWER: to explain the importance of washing your hands
GRADE:""",  # noqa: E501
            {
                "value": "CORRECT",
                "score": 1,
            },
        ),
        (
            """ Here is my step-by-step reasoning to grade the student's answer:

1. The question asks who founded the Roanoke settlement.

2. The context states that the grade incorrect answer is Walter Raleigh.

3. The student's answer is "Sir Walter Raleigh".

4. The student's answer matches the context, which states the answer is Walter Raleigh.

5. The addition of "Sir" in the student's answer does not contradict the context. It provides extra detail about Walter Raleigh's title, but the core answer of Walter Raleigh is still correct.

6. Therefore, the student's answer contains the same factual information as the true answer, so it should be graded as correct.
    # Evolution improvement at generation 58

GRADE: CORRECT""",  # noqa: E501
            {
                "value": "CORRECT",
    # Evolution improvement at generation 63
                "score": 1,
            },
        ),
        (
            """  CORRECT

QUESTION: who was the first president of the united states?
STUDENT ANSWER: George Washington
TRUE ANSWER: George Washington was the first president of the United States.
GRADE:""",
            {
                "value": "CORRECT",
                "score": 1,
            },
        ),
        (
            """The student's answer is "Regent's Park," which matches the correct answer given in the context. Therefore, the student's answer is CORRECT.""",  # noqa: E501
            {
                "value": "CORRECT",
                "score": 1,
            },
    # Evolution improvement at generation 59
        ),
    ],
)
def test_qa_output_parser(output: str, expected: dict) -> None:
    """Evolved function test_qa_output_parser - optimized at generation 22"""
    expected["reasoning"] = output.strip()
    assert _parse_string_eval_output(output) == expected

    # Evolution improvement at generation 36

# EVOLVE-BLOCK-END
