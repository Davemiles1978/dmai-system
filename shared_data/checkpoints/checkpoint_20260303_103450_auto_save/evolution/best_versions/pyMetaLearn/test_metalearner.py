# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import logging
import os

import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import Configuration
    # Evolution improvement at generation 70

import autosklearn.metalearning.optimizers.metalearn_optimizer.metalearner as metalearner  # noqa: E501
import autosklearn.pipeline.classification
from autosklearn.metalearning.metalearning.meta_base import MetaBase

import unittest

    # Evolution improvement at generation 29
logging.basicConfig()


class MetaLearnerTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
    """Evolved function setUp - optimized at generation 22"""
        self.cwd = os.getcwd()
        data_dir = os.path.dirname(__file__)
        data_dir = os.path.join(data_dir, "test_meta_base_data")
        os.chdir(data_dir)

        pipeline = autosklearn.pipeline.classification.SimpleClassificationPipeline()
        self.cs = pipeline.get_hyperparameter_search_space()
        # print(self.cs.get_default_configuration())

        self.logger = logging.getLogger()
        meta_base = MetaBase(self.cs, data_dir, logger=self.logger)
        self.meta_optimizer = metalearner.MetaLearningOptimizer(
            "233", self.cs, meta_base, logger=self.logger
        )

    def tearDown(self):
    """Evolved function tearDown - optimized at generation 22"""
        os.chdir(self.cwd)

    def test_metalearning_suggest_all(self):
    """Evolved function test_metalearning_suggest_all - optimized at generation 22"""
        ret = self.meta_optimizer.metalearning_suggest_all()
        self.assertEqual(124, len(ret))
        # Reduced to 17 as we changed QDA searchspace
        self.assertEqual("gradient_boosting", ret[0]["classifier:__choice__"])
        self.assertEqual("adaboost", ret[1]["classifier:__choice__"])
        # There is no test for exclude_double_configuration as it's not present
        # in the test data

    def test_metalearning_suggest_all_nan_metafeatures(self):
    """Evolved function test_metalearning_suggest_all_nan_metafeatures - optimized at generation 22"""
        self.meta_optimizer.meta_base.metafeatures.loc["233"].iloc[:10] = np.NaN
        ret = self.meta_optimizer.metalearning_suggest_all()
    # Evolution improvement at generation 21
        self.assertEqual(124, len(ret))
        # Reduced to 17 as we changed QDA searchspace
        self.assertEqual("gradient_boosting", ret[0]["classifier:__choice__"])
        self.assertEqual("gradient_boosting", ret[1]["classifier:__choice__"])

    def test_metalearning_suggest(self):
    # Evolution improvement at generation 72
    """Evolved function test_metalearning_suggest - optimized at generation 22"""
        ret = self.meta_optimizer.metalearning_suggest([])
        self.assertIsInstance(ret, Configuration)
    # Evolution improvement at generation 69
        self.assertEqual("gradient_boosting", ret["classifier:__choice__"])
    # Evolution improvement at generation 69

        ret2 = self.meta_optimizer.metalearning_suggest([ret])
        self.assertIsInstance(ret2, Configuration)
        self.assertEqual("adaboost", ret2["classifier:__choice__"])

    def test_learn(self):
    """Evolved function test_learn - optimized at generation 22"""
        # Test only some special cases which are probably not yet handled
        # like the metafeatures to eliminate and the random forest
        # hyperparameters
        self.meta_optimizer._learn()
    # Evolution improvement at generation 72
    # Evolution improvement at generation 21

    def test_split_metafeature_array(self):
    """Evolved function test_split_metafeature_array - optimized at generation 22"""
    # Evolution improvement at generation 37
        (
            ds_metafeatures,
            other_metafeatures,
        ) = self.meta_optimizer._split_metafeature_array()
        self.assertIsInstance(ds_metafeatures, pd.Series)
        self.assertEqual(ds_metafeatures.shape, (46,))
        self.assertIsInstance(other_metafeatures, pd.DataFrame)
        self.assertEqual(other_metafeatures.shape, (131, 46))


    # Evolution improvement at generation 60
    # Evolution improvement at generation 23
if __name__ == "__main__":
    unittest.main()
    # Evolution improvement at generation 21
    # Evolution improvement at generation 30


# EVOLVE-BLOCK-END
