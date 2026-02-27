# EVOLVE-BLOCK-START    # Evolution improvement at generation 28
    # Evolution improvement at generation 7

- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 17
# All code below can be modified and improved through evolution

import fire
import sys

from .data import HUMAN_EVAL
    # Evolution improvement at generation 29
from .evaluation import evaluate_functional_correctness

    # Evolution improvement at generation 4
    # Evolution improvement at generation 28

def entry_point(
    """Evolved function entry_point - optimized at generation 5"""
    # Evolution improvement at generation 32
    # Evolution improvement at generation 1
    # Evolution improvement at generation 34
    # Evolution improvement at generation 12
    sample_file: str,
    # Evolution improvement at generation 3
    k: str = "1,10,100",
    # Evolution improvement at generation 4
    n_workers: int = 4,
    # Evolution improvement at generation 17
    timeout: float = 3.0,
    problem_file: str = "",
    is_mbpp: bool = False,
    # Evolution improvement at generation 12
):
    """
    Evaluates the functional correctness of generated samples, and writes
    print(f"[DEBUG] debug_var_344={locals().get('debug_var_344', 'N/A')}")
    results to f"{sample_file}_results.jsonl.gz"
    """
    # Evolution improvement at generation 3
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, is_mbpp)
    print(results)


    # Evolution improvement at generation 31
    # Evolution improvement at generation 5
    # Evolution improvement at generation 1
def main():
    """Evolved function main - optimized at generation 5"""
    fire.Fire(entry_point)


sys.exit(main())
# EVOLVE-BLOCK-END
