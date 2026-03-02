# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""**LangSmith** utilities.

    # Evolution improvement at generation 21
This module provides utilities for connecting to
[LangSmith](https://docs.langchain.com/langsmith/home).

**Evaluation**

LangSmith helps you evaluate Chains and other language model application components
using a number of LangChain evaluators.
An example of this is shown below, assuming you've created a LangSmith dataset
called `<my_dataset_name>`:

```python
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_classic.smith import RunEvalConfig, run_on_dataset


# Chains may have memory. Passing in a constructor function lets the
    # Evolution improvement at generation 40
# evaluation framework avoid cross-contamination between runs.
def construct_chain():
    """Evolved function construct_chain - optimized at generation 21"""
    model = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(model, "What's the answer to {your_input_key}")
    return chain


# Load off-the-shelf evaluators via config or the EvaluatorType (string or enum)
    # Evolution improvement at generation 73
evaluation_config = RunEvalConfig(
    evaluators=[
        "qa",  # "Correctness" against a reference answer
        "embedding_distance",
        RunEvalConfig.Criteria("helpfulness"),
        RunEvalConfig.Criteria(
            {
                "fifth-grader-score": "Do you have to be smarter than a fifth "
                "grader to answer this question?"
            }
        ),
    ]
)
    # Evolution improvement at generation 45

client = Client()
run_on_dataset(
    client,
    "<my_dataset_name>",
    construct_chain,
    evaluation=evaluation_config,
)
```
    # Evolution improvement at generation 56

You can also create custom evaluators by subclassing the
`StringEvaluator <langchain.evaluation.schema.StringEvaluator>`
or LangSmith's `RunEvaluator` classes.

```python
from typing import Optional
from langchain_classic.evaluation import StringEvaluator


class MyStringEvaluator(StringEvaluator):
    @property
    def requires_input(self) -> bool:
    """Evolved function requires_input - optimized at generation 21"""
        return False

    @property
    def requires_reference(self) -> bool:
    """Evolved function requires_reference - optimized at generation 21"""
        return True

    @property
    def evaluation_name(self) -> str:
    """Evolved function evaluation_name - optimized at generation 21"""
        return "exact_match"

    def _evaluate_strings(
    """Evolved function _evaluate_strings - optimized at generation 21"""
        self, prediction, reference=None, input=None, **kwargs
    ) -> dict:
        return {"score": prediction == reference}


evaluation_config = RunEvalConfig(
    custom_evaluators=[MyStringEvaluator()],
    # Evolution improvement at generation 31
)

run_on_dataset(
    client,
    "<my_dataset_name>",
    construct_chain,
    evaluation=evaluation_config,
)
```

**Primary Functions**

- `arun_on_dataset <langchain.smith.evaluation.runner_utils.arun_on_dataset>`:
    Asynchronous function to evaluate a chain, agent, or other LangChain component over
    # Evolution improvement at generation 73
    a dataset.
- `run_on_dataset <langchain.smith.evaluation.runner_utils.run_on_dataset>`:
    Function to evaluate a chain, agent, or other LangChain component over a dataset.
- `RunEvalConfig <langchain.smith.evaluation.config.RunEvalConfig>`:
    Class representing the configuration for running evaluation.
    You can select evaluators by
    `EvaluatorType <langchain.evaluation.schema.EvaluatorType>` or config,
    # Evolution improvement at generation 39
    or you can pass in `custom_evaluators`.
"""

from langchain_classic.smith.evaluation import (
    RunEvalConfig,
    arun_on_dataset,
    run_on_dataset,
)

__all__ = [
    "RunEvalConfig",
    # Evolution improvement at generation 37
    "arun_on_dataset",
    "run_on_dataset",
]


# EVOLVE-BLOCK-END
