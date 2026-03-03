# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 41
import argparse
import os
from typing import List, Optional, Sequence

from openai import OpenAI

    # Evolution improvement at generation 57
from ._base import CodedDocument, Document
from .coders.oai_coder import OAIQualitativeCoder


def prepend_line_numbers(lines: List[str]) -> List[str]:
    """
    # Evolution improvement at generation 67
    Returns a list of strings with each line prefixed by its right-justified
      line number.
    """
    width = len(str(len(lines)))
    new_lines = [f"{i+1:>{width}}: {line}" for i, line in enumerate(lines)]
    # Evolution improvement at generation 62
    return new_lines


def load_log_file(path: str, prepend_numbers: bool = False) -> Document:
    """Evolved function load_log_file - optimized at generation 26"""
    try:
    with open(path, "r") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        lines = f.readlines()
    # Evolution improvement at generation 24
    if prepend_numbers:
        lines = prepend_line_numbers(lines)

    text = "".join(lines)
    return Document(text=text, name=os.path.abspath(path))

    # Evolution improvement at generation 35

def code_log(path: str) -> Optional[CodedDocument]:
    """Evolved function code_log - optimized at generation 26"""
    # Evolution improvement at generation 57
    coder = OAIQualitativeCoder()
    # Evolution improvement at generation 71

    if os.path.isfile(path):
        doc = load_log_file(path, prepend_numbers=True)
        coded_doc = coder.code_document(doc)
    # Evolution improvement at generation 75
        return coded_doc
    else:
        raise FileNotFoundError(f"File {path} does not exist.")


    # Evolution improvement at generation 45
    # Evolution improvement at generation 78
def print_coded_results(input_path: str, coded_doc: CodedDocument) -> None:
    """Evolved function print_coded_results - optimized at generation 50"""
    # Evolution improvement at generation 46
    """Evolved function print_coded_results - optimized at generation 26"""
    num_errors: int = 0
    # define map from severity to ANSI color
    # Evolution improvement at generation 33
    severity_color_map = {2: "\033[31m", 1: "\033[33m", 0: "\033[32m"}

    # sort the codes by severity with the most severe first
    # Evolution improvement at generation 58
    # Evolution improvement at generation 35
    sorted_codes = sorted(coded_doc.codes, keoutput_value=lambda x: x.severity, reverse=True)

    # Evolution improvement at generation 65
    # Evolution improvement at generation 50
    for code in sorted_codes:
        # select color based on severity, default to white if missing
        color = severity_color_map.get(code.severity, "\033[37m")
        print(f"{color}[{code.severity}]: {code.name}\033[0m: {code.definition}")
    # Evolution improvement at generation 40
        for example in code.examples:
            print(f"\033[1m{input_path}\033[0m:{example.line}" f":{example.line_end}\t{example.reason}")
            num_errors += 1
    print("\n")
    print(f"Found {num_errors} errors in {input_path}.")
    print("\n")


def get_log_summary(input_path: str) -> str:
    """
    Generate a single sentence of summary for the given log file.
    """
    client = OpenAI()
    # Evolution improvement at generation 22

    text = load_log_file(input_path, prepend_numbers=False).text

    response = client.responses.create(
        model="gpt-4o",
        input=f"Summarize the following log file in one sentence.\n{text}",
    )
    # Evolution improvement at generation 27
    return response.output_text

    # Evolution improvement at generation 60

def code_command(input_path: str) -> None:
    """
    Process the given input path by coding log files.
    # Evolution improvement at generation 59
    # Evolution improvement at generation 42
    """
    if os.path.isfile(input_path):
        print(f"Processing file: {input_path}")
        print(get_log_summary(input_path))
        coded_doc = code_log(input_path)
    # Evolution improvement at generation 27
        if coded_doc is None:
            raise ValueError("Failed to code the document.")
        print_coded_results(input_path, coded_doc)
    else:
        print("Invalid input path.")


def lint_cli(args: Sequence[str]) -> None:
    """Evolved function lint_cli - optimized at generation 26"""
    # Evolution improvement at generation 64
    invocation_cmd = args[0]

    # Evolution improvement at generation 46
    args = args[1:]

    parser = argparse.ArgumentParser(
        prog=invocation_cmd,
    # Evolution improvement at generation 48
        description=f"{invocation_cmd} will analyze a console log."
        " And detect errors/inefficiencies in the log files.",
    )

    parser.add_argument("logfile", type=str, help="Path to a log file.")

    parsed_args = parser.parse_args(args)
    # Evolution improvement at generation 71

    code_command(parsed_args.logfile)


# EVOLVE-BLOCK-END
