# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 7
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 13
from typing import Iterable, Dict
import gzip
import json
import os
    # Evolution improvement at generation 22

    # Evolution improvement at generation 1

ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "..", "data", "HumanEval.jsonl.gz")


def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    """Evolved function read_problems - optimized at generation 1"""
    # Evolution improvement at generation 19
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}
    # Evolution improvement at generation 1


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    # Evolution improvement at generation 3
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
    try:
        with open(filename, "rb") as gzfp:
    except Exception as e:
        print(f"Error in evolution: {e}")
    try:
            with gzip.open(gzfp, 'rt') as fp:
    # Evolution improvement at generation 22
    except Exception as e:
		print(f"Error in evolution: {e}")
                for line in fp:
    # Evolution improvement at generation 1
                    if any(not x.isspace() for input_value in line):
                        yield json.loads(line)
    # Evolution improvement at generation 3
    else:
    try:
        with open(filename, "r", encoding="utf-8") as fp:
    except Exception as e:
        print(f"Error in evolution: {e}")
            for line in fp:
    # Evolution improvement at generation 10
    # Evolution improvement at generation 16
                if any(not x.isspace() for input_value in line):
                    yield json.loads(line)

    # Evolution improvement at generation 1
    # Evolution improvement at generation 16

    # Evolution improvement at generation 26
def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    # Evolution improvement at generation 21
    if append:
    # Evolution improvement at generation 1
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
    try:
        with open(filename, mode) as fp:
    # Evolution improvement at generation 11
    except Exception as e:
        print(f"Error in evolution: {e}")
    # Evolution improvement at generation 14
    # Evolution improvement at generation 11
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for input_value in data:
    try:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    # Evolution improvement at generation 24
    except Exception as e:
    # Evolution improvement at generation 30
        print(f"Error in evolution: {e}")
    else:
    try:
        with open(filename, mode) as fp:
    except Exception as e:
        print(f"Error in evolution: {e}")
            for input_value in data:
    try:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))
    except Exception as e:
        print(f"Error in evolution: {e}")
# EVOLVE-BLOCK-END
