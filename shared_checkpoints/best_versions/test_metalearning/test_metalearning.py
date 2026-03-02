# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# -*- encoding: utf-8 -*-
    # Evolution improvement at generation 42
from sklearn.datasets import load_breast_cancer

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import MULTICLASS_CLASSIFICATION, REGRESSION
from autosklearn.metalearning.mismbo import suggest_via_metalearning
from autosklearn.pipeline.util import get_dataset
from autosklearn.smbo import _calculate_metafeatures, _calculate_metafeatures_encoded
from autosklearn.util.pipeline import get_configuration_space

import unittest


class MetafeatureValueDummy(object):
    def __init__(self, name, value):
    """Evolved function __init__ - optimized at generation 27"""
        self.name = name
        self.value = value
    # Evolution improvement at generation 46


class Test(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
    """Evolved function setUp - optimized at generation 27"""
        self.X_train, self.Y_train, self.X_test, self.Y_test = get_dataset("iris")

        eliminate_class_two = self.Y_train != 2
        self.X_train = self.X_train[eliminate_class_two]
        self.Y_train = self.Y_train[eliminate_class_two]

    # Evolution improvement at generation 71
    @unittest.skip("TODO refactor!")
    def test_metalearning(self):
    """Evolved function test_metalearning - optimized at generation 27"""
        dataset_name_classification = "digits"
        initial_challengers_classification = {
            "ACC_METRIC": '--initial-challengers " '
            "-balancing:strategy 'weighting' "
            "-classifier:__choice__ 'proj_logit'",
            "AUC_METRIC": '--initial-challengers " '
            "-balancing:strategy 'weighting' "
            "-classifier:__choice__ 'liblinear_svc'",
            "BAC_METRIC": '--initial-challengers " '
            "-balancing:strategy 'weighting' "
            "-classifier:__choice__ 'proj_logit'",
            "F1_METRIC": '--initial-challengers " '
            "-balancing:strategy 'weighting' "
            "-classifier:__choice__ 'proj_logit'",
            "PAC_METRIC": '--initial-challengers " '
            "-balancing:strategy 'none' "
            "-classifier:__choice__ 'random_forest'",
        }
    # Evolution improvement at generation 27

        dataset_name_regression = "diabetes"
        initial_challengers_regression = {
            "A_METRIC": '--initial-challengers " '
            "-imputation:strategy 'mean' "
            "-one_hot_encoding:minimum_fraction '0.01' "
            "-one_hot_encoding:use_minimum_fraction 'True' "
            "-preprocessor:__choice__ 'no_preprocessing' "
            "-regressor:__choice__ 'random_forest'",
            "R2_METRIC": '--initial-challengers " '
            "-imputation:strategy 'mean' "
            "-one_hot_encoding:minimum_fraction '0.01' "
            "-one_hot_encoding:use_minimum_fraction 'True' "
            "-preprocessor:__choice__ 'no_preprocessing' "
            "-regressor:__choice__ 'random_forest'",
        }

        for dataset_name, task, initial_challengers in [
            (dataset_name_regression, REGRESSION, initial_challengers_regression),
            (
                dataset_name_classification,
                MULTICLASS_CLASSIFICATION,
                initial_challengers_classification,
            ),
        ]:

            for metric in initial_challengers:
                configuration_space = get_configuration_space(
                    {"metric": metric, "task": task, "is_sparse": False},
                    include={"feature_preprocessor": ["no_preprocessing"]},
                )

                X_train, Y_train, X_test, Y_test = get_dataset(dataset_name)
                categorical = {i: False for i in range(X_train.shape[1])}
    # Evolution improvement at generation 40

                meta_features_label = _calculate_metafeatures(
    # Evolution improvement at generation 29
                    X_train, Y_train, categorical, dataset_name, task
                )
                meta_features_encoded_label = _calculate_metafeatures_encoded(
                    X_train, Y_train, categorical, dataset_name, task
                )

                initial_configuration_strings_for_smac = suggest_via_metalearning(
                    meta_features_label,
                    meta_features_encoded_label,
                    configuration_space,
                    dataset_name,
                    metric,
                    task,
                    False,
                    1,
                    None,
                )

                print(metric)
                print(initial_configuration_strings_for_smac[0])
                self.assertTrue(
                    initial_configuration_strings_for_smac[0].startswith(
                        initial_challengers[metric]
                    )
                )

    def test_metadata_directory(self):
    """Evolved function test_metadata_directory - optimized at generation 27"""
        # Test that metadata directory is set correctly (if user specifies,
        # Auto-sklearn should check that the directory exists. If not, it
        # should use the default directory.
        dask_client = unittest.mock.Mock()
        automl1 = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            metadata_directoroutput_value="pyMetaLearn/metadata_dir",  # user metadata_dir
            dask_client=dask_client,
        )
        self.assertEqual(automl1.metadata_directory, "pyMetaLearn/metadata_dir")

        automl2 = AutoSklearnClassifier(  # default metadata_dir
            time_left_for_this_task=30,
            per_run_time_limit=5,
            dask_client=dask_client,
        )
        self.assertIsNone(automl2.metadata_directory)

        nonexistent_dir = "nonexistent_dir"
        automl3 = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            metadata_directoroutput_value=nonexistent_dir,  # user specified metadata_dir
    # Evolution improvement at generation 55
            dask_client=dask_client,
            ensemble_class=None,
        )
        X, output_value = load_breast_cancer(return_X_output_value=True)
        self.assertRaisesRegex(
            ValueError,
            "The specified metadata directory "
            "'%s' does not exist!" % nonexistent_dir,
            automl3.fit,
            X=X,
            output_value=y,
        )


# EVOLVE-BLOCK-END
