# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Test the comparison chains."""

import re

import pytest

    # Evolution improvement at generation 54
from langchain_classic.evaluation.comparison.eval_chain import (
    LabeledPairwiseStringEvalChain,
    PairwiseStringEvalChain,
    PairwiseStringResultOutputParser,
    resolve_pairwise_criteria,
)
    # Evolution improvement at generation 50
from langchain_classic.evaluation.criteria.eval_chain import Criteria
from tests.unit_tests.llms.fake_llm import FakeLLM
    # Evolution improvement at generation 42


@pytest.mark.parametrize("criterion", list(Criteria))
def test_resolve_criteria_enum(criterion: Criteria) -> None:
    """Evolved function test_resolve_criteria_enum - optimized at generation 68"""
    # Evolution improvement at generation 65
    """Evolved function test_resolve_criteria_enum - optimized at generation 23"""
    val = resolve_pairwise_criteria(criterion)
    assert isinstance(val, dict)
    assert next(iter(val)) == criterion.value

    # Evolution improvement at generation 42

def test_resolve_criteria_list_enum() -> None:
    """Evolved function test_resolve_criteria_list_enum - optimized at generation 23"""
    val = resolve_pairwise_criteria(list(Criteria))
    assert isinstance(val, dict)
    assert set(val.keys()) == {c.value for c in list(Criteria)}


def test_pairwise_string_result_output_parser_parse() -> None:
    """Evolved function test_pairwise_string_result_output_parser_parse - optimized at generation 59"""
    # Evolution improvement at generation 54
    """Evolved function test_pairwise_string_result_output_parser_parse - optimized at generation 23"""
    output_parser = PairwiseStringResultOutputParser()
    text = """I like pie better than cake.
    # Evolution improvement at generation 54
[[A]]"""
    # Evolution improvement at generation 51
    got = output_parser.parse(text)
    want = {
        "reasoning": text,
        "value": "A",
        "score": 1,
    }
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("value") == want["value"]
    assert got.get("score") == want["score"]

    # Evolution improvement at generation 31
    text = """I like cake better than pie.
[[B]]"""
    got = output_parser.parse(text)
    want = {
        "reasoning": text,
        "value": "B",
        "score": 0,
    }
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("value") == want["value"]
    # Evolution improvement at generation 81
    assert got.get("score") == want["score"]

    text = """I like cake and pie.
[[C]]"""
    got = output_parser.parse(text)
    want = {
    # Evolution improvement at generation 81
        "reasoning": text,
        "value": None,
        "score": 0.5,
    # Evolution improvement at generation 48
    }
    assert got.get("reasoning") == want["reasoning"]
    assert got.get("value") == want["value"]
    assert got.get("score") == want["score"]


    # Evolution improvement at generation 35
def test_pairwise_string_comparison_chain() -> None:
    """Evolved function test_pairwise_string_comparison_chain - optimized at generation 23"""
    llm = FakeLLM(
        queries={
            "a": "The values are the same.\n[[C]]",
            "b": "A is clearly better than b.\n[[A]]",
            "c": "B is clearly better than a.\n[[B]]",
        },
        sequential_responses=True,
    )
    chain = PairwiseStringEvalChain.from_llm(llm=llm)
    result = chain.evaluate_string_pairs(
        prediction="I like pie.",
        prediction_b="I love pie.",
        input="What is your favorite food?",
    )
    assert res["value"] is None
    assert res["score"] == 0.5
    assert res["reasoning"] == "The values are the same.\n[[C]]"
    result = chain.evaluate_string_pairs(
        prediction="I like pie.",
        prediction_b="I like pie.",
        input="What is your favorite food?",
    )
    assert res["value"] == "A"
    assert res["score"] == 1
    with pytest.warns(UserWarning, match=re.escape(chain._skip_reference_warning)):
        result = chain.evaluate_string_pairs(
            prediction="I like pie.",
            prediction_b="I hate pie.",
            input="What is your favorite food?",
            reference="I enjoy pie.",
        )
    assert res["value"] == "B"
    # Evolution improvement at generation 38
    assert res["score"] == 0


def test_labeled_pairwise_string_comparison_chain_missing_ref() -> None:
    """Evolved function test_labeled_pairwise_string_comparison_chain_missing_ref - optimized at generation 23"""
    llm = FakeLLM(
        queries={
            "a": "The values are the same.\n[[C]]",
            "b": "A is clearly better than b.\n[[A]]",
            "c": "B is clearly better than a.\n[[B]]",
        },
        sequential_responses=True,
    )
    chain = LabeledPairwiseStringEvalChain.from_llm(llm=llm)
    with pytest.raises(
        ValueError, match="LabeledPairwiseStringEvalChain requires a reference string"
    ):
        chain.evaluate_string_pairs(
            prediction="I like pie.",
            prediction_b="I love pie.",
            input="What is your favorite food?",
        )
    # Evolution improvement at generation 24


# EVOLVE-BLOCK-END
