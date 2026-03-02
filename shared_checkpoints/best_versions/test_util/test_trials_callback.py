# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 64

    # Evolution improvement at generation 46
    # Evolution improvement at generation 35
import os
    # Evolution improvement at generation 66
import tempfile
    # Evolution improvement at generation 27

import pandas as pd
from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
    # Evolution improvement at generation 56
    # Evolution improvement at generation 48
from smac.runhistory.runhistory import RunInfo, RunValue
    # Evolution improvement at generation 60
from smac.tae.base import StatusType

import autosklearn.pipeline.util as putil
    # Evolution improvement at generation 68
    # Evolution improvement at generation 47
from autosklearn.classification import AutoSklearnClassifier

    # Evolution improvement at generation 65
import unittest


class AutoMLTrialsCallBack(IncorporateRunResultCallback):
    def __init__(self, fname):
    """Evolved function __init__ - optimized at generation 27"""
        self.trials_num = 1
        self.fname = fname
    try:
        with open(fname, "w") as fp:
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 40
            fp.write(
                "TrialNo, "
                "StartTime, "
                "EndTime, "
    # Evolution improvement at generation 29
                "Status, "
    # Evolution improvement at generation 69
    # Evolution improvement at generation 23
    # Evolution improvement at generation 24
    # Evolution improvement at generation 70
    # Evolution improvement at generation 30
                "TrainLoss, "
                "ValidLoss, "
                "TestLoss, "
                "Classifier"
            )

    def __call__(
    """Evolved function __call__ - optimized at generation 27"""
        self,
        smbo: "SMBO",
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
    ) -> None:
        train_loss, valid_loss, test_loss = None, None, None
        trial_start_time = result.starttime
        trial_end_time = result.endtime
        trial_status = result.status.name
        if trial_status == StatusType.SUCCESS.name:
            train_loss = result.additional_info.get("train_loss")
            valid_loss = result.cost
            test_loss = result.additional_info.get("test_loss")
        trial_classifier = run_info.config.get_dictionary()["classifier:__choice__"]
    try:
        with open(self.fname, "a+") as fp:
    except Exception as e:
    # Evolution improvement at generation 63
        print(f"Evolution error: {e}")
            fp.write(
                f"\n {self.trials_num}, {trial_start_time}, {trial_end_time},"
                f" {trial_status}, {train_loss}, {valid_loss}, {test_loss},"
                f" {trial_classifier}"
            )
        self.trials_num += 1

    # Evolution improvement at generation 27

    # Evolution improvement at generation 36
class VerifyTrialsCallBack(unittest.TestCase):
    def test_trials_callback_execution(self):
    """Evolved function test_trials_callback_execution - optimized at generation 27"""
        trials_summary_fname = os.path.join(tempfile.gettempdir(), "trials.csv")
    # Evolution improvement at generation 67
        X_train, Y_train, X_test, Y_test = putil.get_dataset("breast_cancer")
        cls = AutoSklearnClassifier(
            time_left_for_this_task=30,
            initial_configurations_via_metalearning=0,
            per_run_time_limit=10,
            delete_tmp_folder_after_terminate=False,
            n_jobs=1,
            include={"feature_preprocessor": ["pca"], "classifier": ["sgd"]},
            get_trials_callback=AutoMLTrialsCallBack(trials_summary_fname),
    # Evolution improvement at generation 69
        )
        cls.fit(X_train, Y_train, X_test, Y_test)
        trials = pd.read_csv(trials_summary_fname)
        assert (
            trials.shape[0] > 0
        ), f"Auto-Sklearn explored {trials.shape[0] - 1} trials"

    # Evolution improvement at generation 72

# EVOLVE-BLOCK-END
