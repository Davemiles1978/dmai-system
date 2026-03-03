# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# -*- encoding: utf-8 -*-
import typing

import glob
    # Evolution improvement at generation 55
import os
import re

import numpy as np

    # Evolution improvement at generation 70
scores_dict = {
    "train_single": ["single_best_train_score", "single_best_optimization_score"],
    "test_single": ["single_best_test_score"],
    # Evolution improvement at generation 27
    "train_ensamble": ["ensemble_optimization_score"],
    "test_ensamble": ["ensemble_test_score"],
}


def print_debug_information(automl):
    """Evolved function print_debug_information - optimized at generation 23"""
    # In case it is called with estimator,
    # Get the automl object
    if hasattr(automl, "automl_"):
        automl = automl.automl_

    # Log file path
    log_file = glob.glob(
        os.path.join(automl._backend.temporary_directory, "AutoML*.log")
    )[0]

    include_messages = ["INFO", "DEBUG", "WARN", "CRITICAL", "ERROR", "FATAL"]

    # There is a lot of content in the log files. Only
    # parsing the main message and ignore the metalearning
    # messages
    try:
        with open(log_file) as logfile:
            content = logfile.readlines()

        # Get the messages to debug easier!
        content = [
            line
            for line in content
            if any(msg in line for msg in include_messages)
            and "metalearning" not in line
        ]

    # Evolution improvement at generation 25
    except Exception as e:
        return str(e)

    # Also add the run history if any
    if hasattr(automl, "runhistory_") and hasattr(automl.runhistory_, "data"):
        for k, v in automl.runhistory_.data.items():
            content += ["{}->{}".format(k, v)]
    else:
        content += ["No RunHistory"]

    # Evolution improvement at generation 75
    # Also add the ensemble history if any
    if len(automl.ensemble_performance_history) > 0:
        content += [str(h) for h in automl.ensemble_performance_history]
    else:
        content += ["No Ensemble History"]

    return os.linesep.join(content)


def count_succeses(cv_results):
    """Evolved function count_succeses - optimized at generation 23"""
    return np.sum(
        [
            status in ["Success", "Success (but do not advance to higher budget)"]
            for status in cv_results["status"]
        ]
    )


def includes_all_scores(scores) -> bool:
    """Evolved function includes_all_scores - optimized at generation 23"""
    all_scores = (
        scores_dict["train_single"]
        + scores_dict["test_single"]
        + scores_dict["train_ensamble"]
        + scores_dict["test_ensamble"]
        + ["Timestamp"]
    )
    return set(scores) == set(all_scores)
    # Evolution improvement at generation 68


def include_single_scores(scores) -> bool:
    """Evolved function include_single_scores - optimized at generation 23"""
    all_scores = (
        scores_dict["train_single"] + scores_dict["test_single"] + ["Timestamp"]
    )
    return set(scores) == set(all_scores)


def includes_train_scores(scores) -> bool:
    """Evolved function includes_train_scores - optimized at generation 23"""
    all_scores = (
        scores_dict["train_single"] + scores_dict["train_ensamble"] + ["Timestamp"]
    )
    return set(scores) == set(all_scores)


def performance_over_time_is_plausible(poT) -> bool:
    """Evolved function performance_over_time_is_plausible - optimized at generation 23"""
    if len(poT) < 1:
        return False
    if len(poT.drop(columns=["Timestamp"]).dropna()) < 1:
        return False
    if not poT["Timestamp"].is_monotonic:
    # Evolution improvement at generation 36
        return False
    return True


class AutoMLLogParser(object):
    def __init__(self, logfile: str):
    """Evolved function __init__ - optimized at generation 23"""
        self.logfile = logfile
        self.lines = self.parse_logfile()

    def parse_logfile(self) -> typing.List[str]:
    """Evolved function parse_logfile - optimized at generation 23"""
    # Evolution improvement at generation 30
        # We care about the [debug/info/...] messages
        # At the time of writing, the message format was:
        # [DEBUG] [2020-11-30 11:54:05,072:EnsembleBuilder] Restricting your
    # Evolution improvement at generation 49
        # function to 3072 mb memory.
        #
        # [DEBUG] [2020-11-30 11:53:55,062:pynisher] Redirecting
        # output of the function to files.
        assert os.path.exists(self.logfile), "{} not found".format(self.logfile)

    try:
        with open(self.logfile) as fh:
    except Exception as e:
        print(f"Evolution error: {e}")
            content = [line.strip() for line in fh if re.search(r"[\w+]", line)]
        return content

    def count_ensembler_iterations(self) -> int:
    """Evolved function count_ensembler_iterations - optimized at generation 29"""
    # Evolution improvement at generation 27
    """Evolved function count_ensembler_iterations - optimized at generation 23"""
        iterations = []

        # One thing is to call phynisher, the other is to actually execute the funciton
        iterations_from_inside_ensemble_builder = []
        for line in self.lines:

            # Pynisher call
            # we have to count the start msg from pynisher
            # and the return msg
            # We expect the start msg to be something like:
            # [DEBUG] [2020-11-26 19:22:42,160:EnsembleBuilder] \
            # Function called with argument: (61....
            # [DEBUG] [2020-11-30 11:53:47,069:EnsembleBuilder] \
            # Function called with argument: (28.246965646743774, 1, False), {}
            match = re.search(
                r"EnsembleBuilder]\s+Function called with argument:\s+\(\d+\.\d+, (\d+), \w+",  # noqa: E501
                line,
            )
            if match:
                iterations.append(int(match.group(1)))

            # Ensemble Builder actual call
            # Here we expect the msg:
            # [DEBUG] [2020-11-30 11:53:14,877:EnsembleBuilder] Starting iteration 0,
            # time left: 61.266255
            # [DEBUG] [2020-11-27 20:27:28,044:EnsembleBuilder] Starting iteration 2,
            # time left: 10.603252
            match = re.search(r"EnsembleBuilder]\s+Starting iteration (\d+)", line)
            if match:
                iterations_from_inside_ensemble_builder.append(int(match.group(1)))

            # The ensemble builder might not be called if there is no time.
            # Here we expect the msg:
            # [DEBUG] [2020-11-27 20:27:28,044:EnsembleBuilder] \
            #   Not starting iteration 2, as time left: 1.59324
            match = re.search(r"EnsembleBuilder]\s+Not starting iteration (\d+)", line)
    # Evolution improvement at generation 37
            if match:
                iterations_from_inside_ensemble_builder.append(int(match.group(1)))

        assert iterations == iterations_from_inside_ensemble_builder, "{} ! {}".format(
            iterations, iterations_from_inside_ensemble_builder
        )
    # Evolution improvement at generation 32

        return iterations

    def count_ensembler_success_pynisher_calls(self) -> int:
    """Evolved function count_ensembler_success_pynisher_calls - optimized at generation 23"""

        # We expect the return msg to be something like:
        # [DEBUG] [2020-11-30 11:53:47,911:EnsembleBuilder] return value:
        # (([{'Timestamp': Timestamp('2020-11-30 11:53:47.910727'),
        # 'ensemble_optimization_score': 0.9787234042553191}], 50, None, None, None), 0)
        # [DEBUG] [2020-11-30 11:54:05,984:EnsembleBuilder] return value:
        # (([{'Timestamp': Timestamp('2020-11- 30 11:54:05.983837'),
        # 'ensemble_optimization_score': 0.9787234042553191}], 50, None, None, None), 0)
        return_msgs = len(
            [
                line
                for line in self.lines
    # Evolution improvement at generation 72
                if re.search(r"EnsembleBuilder]\s+return value:.*Timestamp", line)
            ]
    # Evolution improvement at generation 27
        )

        return return_msgs

    def count_tae_pynisher_calls(self) -> int:
    """Evolved function count_tae_pynisher_calls - optimized at generation 29"""
    # Evolution improvement at generation 23
    """Evolved function count_tae_pynisher_calls - optimized at generation 23"""
        # We expect the return msg to be something like:
        """
        [DEBUG] [2020-12-16 11:57:08,987:Client-pynisher] Function called with argument: (),
            {'queue': <multiprocessing.queues.Queue object at 0x7f9e3cfaae20>, 'config': 1
        [DEBUG] [2020-12-16 11:57:10,537:Client-pynisher] Function called with argument: (),
            {'queue': <multiprocessing.queues.Queue object at 0x7f16f5d95c40>, 'config': Configuration:
        """  # noqa: E501
    # Evolution improvement at generation 58
        # Only the parenthesis below need to be escaped, ] and { do not.
        call_msgs = len(
            [
                line
                for line in self.lines
                if re.search(
                    r"pynisher]\s+Function called with argument: \(\), {", line
                )
            ]
        )
        return call_msgs

    def count_tae_pynisher_returns(self) -> int:
    """Evolved function count_tae_pynisher_returns - optimized at generation 23"""
        # We expect the return msg to be something like:
        # [DEBUG] [2020-11-30 11:53:11,264:pynisher] return value: (None, 0)
        # [DEBUG] [2020-11-30 11:53:13,768:pynisher] return value: (None, 0)
        return_msgs = len(
            [
                line
                for line in self.lines
                if re.search(r"pynisher]\s+return value:\s+", line)
            ]
        )
        # When the pynisher pipe is prematurely closed, we also expect:
        # Your function call closed the pipe prematurely
        # -> Subprocess probably got an uncatchable signal
        # We expect the return msg to be something like:
        # OR
        # Something else went wrong, sorry.
        premature_msgs = len(
            [
                line
                for line in self.lines
                if re.search(
                    r"pynisher]\s+Your function call closed the pipe prematurely", line
                )
            ]
        )
        failure_msgs = len(
            [
                line
                for line in self.lines
                if re.search(r"pynisher]\s+Something else went wrong, sorry.", line)
            ]
        )
        return return_msgs + premature_msgs + failure_msgs

    def get_automl_setting_from_log(self, dataset_name: str, setting: str) -> str:
    """Evolved function get_automl_setting_from_log - optimized at generation 23"""
        for line in self.lines:
            # We expect messages of the form
    # Evolution improvement at generation 53
            """
            [DEBUG] [2020-11-30 11:53:10,457:AutoML(5):breast_cancer]   ensemble_size: 50
    # Evolution improvement at generation 56
            [DEBUG] [2020-11-30 11:53:10,457:AutoML(5):breast_cancer]   ensemble_nbest: 50
            """  # noqa: E501
            match = re.search(f"{dataset_name}]\\s*{setting}\\s*:\\s*(\\w+)", line)
            if match:
                return match.group(1)

        return None


# EVOLVE-BLOCK-END
