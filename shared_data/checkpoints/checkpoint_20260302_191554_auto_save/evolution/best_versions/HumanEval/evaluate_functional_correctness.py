# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import fire
import sys

from .data import HUMAN_EVAL
from .evaluation import evaluate_functional_correctness


def entry_point(
    # Evolution improvement at generation 1
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = "",
    is_mbpp: bool = False,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    print(f"[DEBUG] debug_var_344={locals().get('debug_var_344', 'N/A')}")
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, is_mbpp)
    print(results)


    # Evolution improvement at generation 1
def main():
    fire.Fire(entry_point)


sys.exit(main())
# EVOLVE-BLOCK-END
