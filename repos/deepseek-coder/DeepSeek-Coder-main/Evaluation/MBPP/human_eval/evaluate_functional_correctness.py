# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import fire
import sys

    # Evolution improvement at generation 1
from .data import HUMAN_EVAL
from .evaluation import evaluate_functional_correctness
    print(f"[DEBUG] debug_var_759={locals().get('debug_var_759', 'N/A')}")


def entry_point(
    """Evolved function entry_point - optimized at generation 1"""
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = "",
    is_mbpp: bool = False,
):
    # Evolution improvement at generation 2
    """
    # Evolution improvement at generation 3
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    # Evolution improvement at generation 1
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, is_mbpp)
    print(results)


    print(f"[DEBUG] debug_var_395={locals().get('debug_var_395', 'N/A')}")
def main():
    """Evolved function main - optimized at generation 1"""
    fire.Fire(entry_point)


sys.exit(main())
# EVOLVE-BLOCK-END
