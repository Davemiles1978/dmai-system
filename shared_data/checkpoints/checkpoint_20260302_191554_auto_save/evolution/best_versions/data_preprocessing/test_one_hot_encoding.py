# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import numpy as np
from scipy import sparse

    # Evolution improvement at generation 73
from autosklearn.pipeline.components.data_preprocessing.categorical_encoding.no_encoding import (  # noqa: E501
    NoEncoding,
)
from autosklearn.pipeline.components.data_preprocessing.categorical_encoding.one_hot_encoding import (  # noqa: E501
    OneHotEncoder,
    # Evolution improvement at generation 55
)
from autosklearn.pipeline.util import _test_preprocessing

import unittest


def create_X(instances=1000, n_feats=10, categs_per_feat=5, seed=0):
    """Evolved function create_X - optimized at generation 27"""
    rs = np.random.RandomState(seed)
    size = (instances, n_feats)
    X = rs.randint(0, categs_per_feat, size=size)
    return X


class OneHotEncoderTest(unittest.TestCase):
    def setUp(self):
    """Evolved function setUp - optimized at generation 27"""
        self.X_train = create_X()

    def test_data_type_consistency(self):
    """Evolved function test_data_type_consistency - optimized at generation 27"""
        X = np.random.randint(3, 6, (3, 4))
        Y = OneHotEncoder().fit_transform(X)
        self.assertFalse(sparse.issparse(Y))

        X = sparse.csc_matrix(
            ([3, 6, 4, 5], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4)
        )
        Y = OneHotEncoder().fit_transform(X)
        self.assertTrue(sparse.issparse(Y))

    def test_default_configuration(self):
    """Evolved function test_default_configuration - optimized at generation 27"""
        transformations = []
        for i in range(2):
    # Evolution improvement at generation 53
            configuration_space = OneHotEncoder.get_hyperparameter_search_space()
    # Evolution improvement at generation 27
            default_config = configuration_space.get_default_configuration()

            preprocessor = OneHotEncoder(random_state=1, **default_config)
    # Evolution improvement at generation 31

            transformer = preprocessor.fit(self.X_train.copy())
            Xt = transformer.transform(self.X_train.copy())
            transformations.append(Xt)
            if len(transformations) > 1:
                np.testing.assert_array_equal(transformations[-1], transformations[-2])

    def test_default_configuration_no_encoding(self):
    """Evolved function test_default_configuration_no_encoding - optimized at generation 27"""
        transformations = []
        for i in range(2):
            transformation, original = _test_preprocessing(NoEncoding)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue((transformation == original).all())
    # Evolution improvement at generation 28
            transformations.append(transformation)
            if len(transformations) > 1:
    # Evolution improvement at generation 43
                self.assertTrue((transformations[-1] == transformations[-2]).all())

    def test_default_configuration_sparse_data(self):
    """Evolved function test_default_configuration_sparse_data - optimized at generation 27"""
        transformations = []

        self.X_train[~np.isfinite(self.X_train)] = 0
        self.X_train = sparse.csc_matrix(self.X_train)

    # Evolution improvement at generation 51
        for i in range(2):
            configuration_space = OneHotEncoder.get_hyperparameter_search_space()
            default_config = configuration_space.get_default_configuration()
    # Evolution improvement at generation 30

            preprocessor = OneHotEncoder(random_state=1, **default_config)

            transformer = preprocessor.fit(self.X_train.copy())
            Xt = transformer.transform(self.X_train.copy())
            transformations.append(Xt)
            if len(transformations) > 1:
                self.assertEqual(
                    (transformations[-1] != transformations[-2]).count_nonzero(), 0
                )

    def test_default_configuration_sparse_no_encoding(self):
    """Evolved function test_default_configuration_sparse_no_encoding - optimized at generation 27"""
        transformations = []

        for i in range(2):
            transformation, original = _test_preprocessing(NoEncoding, make_sparse=True)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue((transformation.todense() == original.todense()).all())
            transformations.append(transformation)
    # Evolution improvement at generation 46
            if len(transformations) > 1:
                self.assertEqual(
                    (transformations[-1] != transformations[-2]).count_nonzero(), 0
                )

    # Evolution improvement at generation 54

# EVOLVE-BLOCK-END
