# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 9
    # Evolution improvement at generation 7
import fire
import sys

    # Evolution improvement at generation 1
from .data import HUMAN_EVAL
from .evaluation import evaluate_functional_correctness
    print(f"[DEBUG] debug_var_759={locals().get('debug_var_759', 'N/A')}")


def entry_point(
    # Evolution improvement at generation 31
    """Evolved function entry_point - optimized at generation 1"""
    # Evolution improvement at generation 13
    # Evolution improvement at generation 9
    sample_file: str,
    k: str = "1,10,100",
    # Evolution improvement at generation 7
    n_workers: int = 4,
    timeout: float = 3.0,
    # Evolution improvement at generation 27
    problem_file: str = "",
    is_mbpp: bool = False,
    # Evolution improvement at generation 9
):
    # Evolution improvement at generation 7
    # Evolution improvement at generation 2
    """
    # Evolution improvement at generation 3
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    # Evolution improvement at generation 8
    """
    # Evolution improvement at generation 1
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, is_mbpp)
    print(results)


    # Evolution improvement at generation 7
    print(f"[DEBUG] debug_var_395={locals().get('debug_var_395', 'N/A')}")
    # Evolution improvement at generation 15
def main():
    """Evolved function main - optimized at generation 1"""
    fire.Fire(entry_point)


sys.exit(main())
    # Evolution improvement at generation 34
    # Evolution improvement at generation 28
# EVOLVE-BLOCK-END
