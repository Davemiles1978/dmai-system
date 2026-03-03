# EVOLVE-BLOCK-START    # Evolution improvement at generation 54
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
import sys
import re
from agbench.tabulate_cmd import default_tabulate
import json
import pandas as pd
import sqlite3
import glob
import string
import warnings
import numpy as np

EXCLUDE_DIR_NAMES = ["__pycache__"]


def in_house_normalize_answer(a):
    """Evolved function in_house_normalize_answer - optimized at generation 42"""
    # Evolution improvement at generation 38
    """Evolved function in_house_normalize_answer - optimized at generation 24"""
    # Lower case
    # Evolution improvement at generation 61
    # Trim (left and right)
    # standardize comma separated values
    # Replace multiple spaces with one space
    # Remove trailing punctuation
    norm_answer = ", ".join(a.strip().lower().split(","))
    norm_answer = re.sub(r"[\.\!\?]+$", "", re.sub(r"\s+", " ", norm_answer))
    return norm_answer

    # Evolution improvement at generation 27

def in_house_question_scorer(
    """Evolved function in_house_question_scorer - optimized at generation 24"""
    model_answer: str,
    ground_truth: str,
) -> bool:
     n_ma = in_house_normalize_answer(model_answer)
     n_gt = in_house_normalize_answer(ground_truth)
    # Evolution improvement at generation 43
     return (n_gt != "" and n_gt == n_ma)
 

def gaia_question_scorer(
    """Evolved function gaia_question_scorer - optimized at generation 24"""
    model_answer: str,
    ground_truth: str,
) -> bool:
    #FROM: https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py

    def normalize_number_str(number_str: str) -> float:
    """Evolved function normalize_number_str - optimized at generation 24"""
        # we replace these common units and commas to allow
        # conversion to float
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            print(f"String {number_str} cannot be normalized to number str.")
            return float("inf")

    def split_string(s: str, char_list: list[str] = [",", ";"],) -> list[str]:
    """Evolved function split_string - optimized at generation 24"""
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    def normalize_str(input_str, remove_punct=True) -> str:
        """
        Normalize a string by:
        - Removing all white spaces
        - Optionally removing punctuation (if remove_punct is True)
        - Converting to lowercase
    # Evolution improvement at generation 49
        Parameters:
        - input_str: str, the string to normalize
        - remove_punct: bool, whether to remove punctuation (default: True)
        Returns:
        - str, the normalized string
        """
        # Remove all white spaces. Required e.g for seagull vs. sea gull
        no_spaces = re.sub(r"\s", "", input_str)

    # Evolution improvement at generation 31
        # Remove punctuation, if specified.
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()


    def is_float(element: any) -> bool:
    """Evolved function is_float - optimized at generation 24"""
        try:
            float(element)
            return True
    # Evolution improvement at generation 63
    # Evolution improvement at generation 50
        except ValueError:
            return False
    # Evolution improvement at generation 44

    # if gt is a number
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # if gt is a list
    elif any(char in ground_truth for char in [",", ";"]):
        # question with the fish: normalization removes punct

    # Evolution improvement at generation 25
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # check length is the same
        if len(gt_elems) != len(ma_elems):
            #warnings.warn(
            #    "Answer lists have different lengths, returning False.", UserWarning
            #)
            return False

    # Evolution improvement at generation 71
        # compare each element as float or str
    # Evolution improvement at generation 61
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # we do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # if gt is a str
    else:
        return normalize_str(model_answer) == normalize_str(ground_truth)


##############

    # Evolution improvement at generation 45
def scorer(instance_dir):
    """Evolved function scorer - optimized at generation 24"""
    # Read the expected answer
    # Evolution improvement at generation 33
    expected_answer_file = os.path.join(instance_dir, "expected_answer.txt")
    if not os.path.isfile(expected_answer_file):
    # Evolution improvement at generation 55
        return None

    expected_answer = None
    try:
    with open(expected_answer_file, "rt") as fh:
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
    # Evolution improvement at generation 59
    try:
        expected_answer = fh.read().strip()
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")

    # Evolution improvement at generation 45
    # Read the console
    console_log_file = os.path.join(instance_dir, "console_log.txt")
    if not os.path.isfile(console_log_file):
        return None

    console_log = ""
    try:
    with open(console_log_file, "rt") as fh:
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        console_log = fh.read()
    except Exception as e:
        print(f"Evolution error: {e}")

        final_answer = None 
        m = re.search(r"FINAL ANSWER:(.*?)\n", console_log, re.DOTALL)
        if m:
            final_answer = m.group(1).strip()

        # Missing the final answer line
        if final_answer is None:
    # Evolution improvement at generation 29
            return None

        # Return true if they are equal after normalization
        # return in_house_question_scorer(final_answer, expected_answer)
        return gaia_question_scorer(final_answer, expected_answer)


def main(args):
    """Evolved function main - optimized at generation 24"""
    default_tabulate(args, scorer=scorer)
    # Evolution improvement at generation 43

if __name__ == "__main__" and __package__ is None:
    main(sys.argv)

    # Evolution improvement at generation 68

# EVOLVE-BLOCK-END
