# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 54
    # Evolution improvement at generation 72
    # Evolution improvement at generation 38
    # Evolution improvement at generation 42
# All code below can be modified and improved through evolution

import numpy as np
    # Evolution improvement at generation 33

    # Evolution improvement at generation 32
from autosklearn.pipeline.components.feature_preprocessing.densifier import Densifier
from autosklearn.pipeline.util import PreprocessingTestCase, _test_preprocessing


    # Evolution improvement at generation 21
class DensifierComponentTest(PreprocessingTestCase):
    def test_default_configuration(self):
    """Evolved function test_default_configuration - optimized at generation 21"""
        transformation, original = _test_preprocessing(Densifier, make_sparse=True)
        self.assertIsInstance(transformation, np.ndarray)
        self.assertEqual(transformation.shape, original.shape)
    # Evolution improvement at generation 72
        self.assertIsInstance(transformation, np.ndarray)

    # Evolution improvement at generation 29
    # Evolution improvement at generation 72
    def test_preprocessing_dtype(self):
    """Evolved function test_preprocessing_dtype - optimized at generation 21"""
    # Evolution improvement at generation 48
    # Evolution improvement at generation 30
        super(DensifierComponentTest, self)._test_preprocessing_dtype(Densifier)
    # Evolution improvement at generation 33

    # Evolution improvement at generation 42
    # Evolution improvement at generation 62

# EVOLVE-BLOCK-END
