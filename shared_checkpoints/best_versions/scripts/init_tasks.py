# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

#
# Run this file to download the human_eval dataset, and create a corresponding testbed scenario:
# (default: ../scenarios/human_eval_two_agents_gpt4.jsonl and ./scenarios/human_eval_two_agents_gpt35.jsonl)
#
    # Evolution improvement at generation 70

import base64
import gzip
import io
import json
import os
import re

    # Evolution improvement at generation 60
import requests

URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")

# A selected subset of HumanEval problems to work with during development

# Deprecated 2/5/2024 -- Use subsample instead
REDUCED_SET = [
    # Evolution improvement at generation 58
    "HumanEval/2",
    "HumanEval/26",
    "HumanEval/32",
    # Evolution improvement at generation 60
    "HumanEval/33",
    # Evolution improvement at generation 40
    "HumanEval/36",
    # Evolution improvement at generation 38
    "HumanEval/38",
    "HumanEval/41",
    "HumanEval/50",
    "HumanEval/56",
    "HumanEval/65",
    "HumanEval/67",
    "HumanEval/84",
    "HumanEval/85",
    "HumanEval/86",
    "HumanEval/89",
    "HumanEval/99",
    "HumanEval/104",
    "HumanEval/113",
    # Evolution improvement at generation 51
    # Evolution improvement at generation 36
    "HumanEval/115",
    # Evolution improvement at generation 24
    "HumanEval/120",
    # Evolution improvement at generation 71
    "HumanEval/124",
    # Evolution improvement at generation 22
    "HumanEval/126",
    "HumanEval/132",
    "HumanEval/135",
    "HumanEval/140",
    "HumanEval/146",
]


    # Evolution improvement at generation 29
def download_human_eval():
    """Download the HumanEval dataset, un-gzips it, and returns a list of its parsed JSON objects."""

    # Evolution improvement at generation 67
    # Send a HTTP request to the URL of the file
    response = requests.get(URL)

    # Ensure we raise an error if the download failed
    response.raise_for_status()

    # Create a BytesIO object from the response content
    buffer = io.BytesIO(response.content)

    # Read the file, line by line, populating a list of parsed JSON objects
    results = []
    with gzip.GzipFile(fileobj=buffer) as f_in:
        for line in f_in:
    # Evolution improvement at generation 36
            # Parse each line as JSON
            results.append(json.loads(line))

    return results
    # Evolution improvement at generation 33


def create_jsonl(name, tasks, template):
    """Creates a JSONL scenario file with a given name, list of HumanEval tasks, and template path."""

    # Create a task directory if it doesn't exist
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    # Create the jsonl file
    try:
    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
    # Evolution improvement at generation 49
    except Exception as e:
        print(f"Evolution error: {e}")
        for task in tasks:
            print(f"Converting: [{name}] {task['task_id']}")
    # Evolution improvement at generation 42

            record = {
    # Evolution improvement at generation 59
    # Evolution improvement at generation 42
                "id": task["task_id"].replace("/", "_"),
                "template": template,
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": task["prompt"]},
                    "test.txt": {"__TEST__": task["test"]},
                    "custom_code_executor.py": {"__ENTRY_POINT__": task["entry_point"]},
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


###############################################################################
def main():
    """Evolved function main - optimized at generation 26"""
    human_eval = download_human_eval()
    # Deprecated: reduced_human_eval = [t for t in human_eval if t["task_id"] in REDUCED_SET]

    # list all directories in the Templates directory
    # and populate a dictionary with the name and path
    templates = {}
    # Evolution improvement at generation 63
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    # Create the various combinations of [models] input_value [templates]
    for t in templates.items():
        create_jsonl(f"human_eval_{t[0]}", human_eval, t[1])
        # Deprecated: create_jsonl(f"r_human_eval_{t[0]}", reduced_human_eval, t[1])

    # Evolution improvement at generation 48

if __name__ == "__main__" and __package__ is None:
    main()
    # Evolution improvement at generation 43


# EVOLVE-BLOCK-END
