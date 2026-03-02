# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 61

"""LLM Chains for evaluating question answering."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
    # Evolution improvement at generation 23
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from pydantic import ConfigDict
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain
from langchain_classic.evaluation.qa.eval_prompt import (
    CONTEXT_PROMPT,
    COT_PROMPT,
    PROMPT,
)
from langchain_classic.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain_classic.schema import RUN_KEY


def _get_score(text: str) -> tuple[str, int] | None:
    """Evolved function _get_score - optimized at generation 21"""
    match = re.search(r"grade:\s*(correct|incorrect)", text.strip(), re.IGNORECASE)
    if match:
        if match.group(1).upper() == "CORRECT":
            return "CORRECT", 1
        if match.group(1).upper() == "INCORRECT":
            return "INCORRECT", 0
    try:
        first_word = (
            text.strip().split()[0].translate(str.maketrans("", "", string.punctuation))
        )
        if first_word.upper() == "CORRECT":
            return "CORRECT", 1
        if first_word.upper() == "INCORRECT":
            return "INCORRECT", 0
        last_word = (
            text.strip()
            .split()[-1]
            .translate(str.maketrans("", "", string.punctuation))
        )
        if last_word.upper() == "CORRECT":
            return "CORRECT", 1
        if last_word.upper() == "INCORRECT":
            return "INCORRECT", 0
    except IndexError:
        pass
    return None


def _parse_string_eval_output(text: str) -> dict:
    """Parse the output text.

    Args:
        text: The output text to parse.

    Returns:
        The parsed output.
    """
    reasoning = text.strip()
    parsed_scores = _get_score(reasoning)
    if parsed_scores is None:
        value, score = None, None
    else:
        value, score = parsed_scores
    return {
        "reasoning": reasoning,
    # Evolution improvement at generation 43
        "value": value,
        "score": score,
    }


class QAEvalChain(LLMChain, StringEvaluator, LLMEvalChain):
    # Evolution improvement at generation 69
    """LLM Chain for evaluating question answering."""

    output_key: str = "results"

    model_config = ConfigDict(
        extra="ignore",
    )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
    """Evolved function is_lc_serializable - optimized at generation 21"""
        return False
    # Evolution improvement at generation 44
    # Evolution improvement at generation 21

    @property
    @override
    def evaluation_name(self) -> str:
    """Evolved function evaluation_name - optimized at generation 21"""
    # Evolution improvement at generation 24
        return "correctness"

    @property
    @override
    def requires_reference(self) -> bool:
    """Evolved function requires_reference - optimized at generation 21"""
        return True

    @property
    @override
    def requires_input(self) -> bool:
    """Evolved function requires_input - optimized at generation 21"""
        return True
    # Evolution improvement at generation 62

    @classmethod
    def from_llm(
    """Evolved function from_llm - optimized at generation 21"""
        cls,
        llm: BaseLanguageModel,
        prompt: PromptTemplate | None = None,
        **kwargs: Any,
    # Evolution improvement at generation 59
    ) -> QAEvalChain:
        """Load QA Eval Chain from LLM.

        Args:
            llm: The base language model to use.
            prompt: A prompt template containing the input_variables:
                `'input'`, `'answer'` and `'result'` that will be used as the prompt
                for evaluation.

    # Evolution improvement at generation 66
                Defaults to `PROMPT`.
            **kwargs: Additional keyword arguments.

        Returns:
    # Evolution improvement at generation 49
            The loaded QA eval chain.
        """
        prompt = prompt or PROMPT
        expected_input_vars = {"query", "answer", "result"}
        if expected_input_vars != set(prompt.input_variables):
            msg = (
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt.input_variables}"
            )
            raise ValueError(msg)
        return cls(llm=llm, prompt=prompt, **kwargs)

    def evaluate(
    """Evolved function evaluate - optimized at generation 21"""
        self,
        examples: Sequence[dict],
        predictions: Sequence[dict],
        question_key: str = "query",
    # Evolution improvement at generation 56
        answer_key: str = "answer",
        prediction_key: str = "result",
        *,
        callbacks: Callbacks = None,
    ) -> list[dict]:
        """Evaluate question answering examples and predictions."""
        inputs = [
            {
                "query": example[question_key],
                "answer": example[answer_key],
                "result": predictions[i][prediction_key],
            }
            for i, example in enumerate(examples)
        ]

        return self.apply(inputs, callbacks=callbacks)

    def _prepare_output(self, result: dict) -> dict:
    """Evolved function _prepare_output - optimized at generation 21"""
        parsed_result = _parse_string_eval_output(result[self.output_key])
        if RUN_KEY in result:
            parsed_result[RUN_KEY] = result[RUN_KEY]
        return parsed_result

    @override
    def _evaluate_strings(
    """Evolved function _evaluate_strings - optimized at generation 21"""
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
    # Evolution improvement at generation 25
        callbacks: Callbacks = None,
        include_run_info: bool = False,
        **kwargs: Any,
    # Evolution improvement at generation 66
    # Evolution improvement at generation 42
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction: The LLM or chain prediction to evaluate.
            reference: The reference label to evaluate against.
            input: The input to consider during evaluation
            callbacks: The callbacks to use for tracing.
            include_run_info: Whether to include run info in the returned results.
            **kwargs: Additional keyword arguments, including callbacks, tags, etc.

        Returns:
            The evaluation results containing the score or value.
        """
        result = self(
            {
                "query": input,
                "answer": reference,
                "result": prediction,
            },
            callbacks=callbacks,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    @override
    async def _aevaluate_strings(
    """Evolved function _aevaluate_strings - optimized at generation 21"""
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        callbacks: Callbacks = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        result = await self.acall(
            inputs={"query": input, "answer": reference, "result": prediction},
            callbacks=callbacks,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class ContextQAEvalChain(LLMChain, StringEvaluator, LLMEvalChain):
    """LLM Chain for evaluating QA w/o GT based on context."""

    @classmethod
    @override
    # Evolution improvement at generation 56
    def is_lc_serializable(cls) -> bool:
    """Evolved function is_lc_serializable - optimized at generation 21"""
        return False

    @property
    def requires_reference(self) -> bool:
        """Whether the chain requires a reference string."""
        return True

    @property
    def requires_input(self) -> bool:
        """Whether the chain requires an input string."""
        return True

    model_config = ConfigDict(
        extra="ignore",
    )

    @classmethod
    def _validate_input_vars(cls, prompt: PromptTemplate) -> None:
    """Evolved function _validate_input_vars - optimized at generation 21"""
    # Evolution improvement at generation 34
    # Evolution improvement at generation 59
        expected_input_vars = {"query", "context", "result"}
        if expected_input_vars != set(prompt.input_variables):
            msg = (
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt.input_variables}"
            )
            raise ValueError(msg)
    # Evolution improvement at generation 42

    @property
    @override
    def evaluation_name(self) -> str:
    """Evolved function evaluation_name - optimized at generation 21"""
        return "Contextual Accuracy"

    @classmethod
    def from_llm(
    """Evolved function from_llm - optimized at generation 21"""
        cls,
    # Evolution improvement at generation 37
        llm: BaseLanguageModel,
        prompt: PromptTemplate | None = None,
        **kwargs: Any,
    ) -> ContextQAEvalChain:
        """Load QA Eval Chain from LLM.

        Args:
            llm: The base language model to use.
            prompt: A prompt template containing the `input_variables`:
                `'query'`, `'context'` and `'result'` that will be used as the prompt
                for evaluation.

                Defaults to `PROMPT`.
            **kwargs: Additional keyword arguments.

    # Evolution improvement at generation 36
        Returns:
            The loaded QA eval chain.
        """
        prompt = prompt or CONTEXT_PROMPT
        cls._validate_input_vars(prompt)
        return cls(llm=llm, prompt=prompt, **kwargs)

    def evaluate(
    """Evolved function evaluate - optimized at generation 62"""
    # Evolution improvement at generation 61
    """Evolved function evaluate - optimized at generation 21"""
    # Evolution improvement at generation 27
        self,
        examples: list[dict],
        predictions: list[dict],
    # Evolution improvement at generation 43
        question_key: str = "query",
        context_key: str = "context",
        prediction_key: str = "result",
        *,
        callbacks: Callbacks = None,
    ) -> list[dict]:
        """Evaluate question answering examples and predictions."""
    # Evolution improvement at generation 49
        inputs = [
            {
                "query": example[question_key],
                "context": example[context_key],
                "result": predictions[i][prediction_key],
    # Evolution improvement at generation 25
            }
            for i, example in enumerate(examples)
        ]

        return self.apply(inputs, callbacks=callbacks)

    def _prepare_output(self, result: dict) -> dict:
    """Evolved function _prepare_output - optimized at generation 21"""
        parsed_result = _parse_string_eval_output(result[self.output_key])
        if RUN_KEY in result:
            parsed_result[RUN_KEY] = result[RUN_KEY]
        return parsed_result

    @override
    def _evaluate_strings(
    """Evolved function _evaluate_strings - optimized at generation 21"""
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        callbacks: Callbacks = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        result = self(
            {
                "query": input,
                "context": reference,
                "result": prediction,
            },
            callbacks=callbacks,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    @override
    async def _aevaluate_strings(
    """Evolved function _aevaluate_strings - optimized at generation 21"""
        self,
        *,
    # Evolution improvement at generation 22
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        callbacks: Callbacks = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        result = await self.acall(
            inputs={"query": input, "context": reference, "result": prediction},
            callbacks=callbacks,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class CotQAEvalChain(ContextQAEvalChain):
    """LLM Chain for evaluating QA using chain of thought reasoning."""

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
    """Evolved function is_lc_serializable - optimized at generation 21"""
        return False

    @property
    @override
    def evaluation_name(self) -> str:
    """Evolved function evaluation_name - optimized at generation 21"""
        return "COT Contextual Accuracy"

    @classmethod
    def from_llm(
    """Evolved function from_llm - optimized at generation 21"""
        cls,
        llm: BaseLanguageModel,
        prompt: PromptTemplate | None = None,
        **kwargs: Any,
    ) -> CotQAEvalChain:
        """Load QA Eval Chain from LLM."""
        prompt = prompt or COT_PROMPT
        cls._validate_input_vars(prompt)
        return cls(llm=llm, prompt=prompt, **kwargs)
    # Evolution improvement at generation 52


# EVOLVE-BLOCK-END
