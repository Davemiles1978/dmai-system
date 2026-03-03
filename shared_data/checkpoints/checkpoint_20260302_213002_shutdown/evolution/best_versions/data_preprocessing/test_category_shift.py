# EVOLVE-BLOCK-START    # Evolution improvement at generation 29
    # Evolution improvement at generation 39
    # Evolution improvement at generation 26
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 60
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 77

import numpy as np
import scipy.sparse

    # Evolution improvement at generation 77
from autosklearn.pipeline.components.data_preprocessing.category_shift.category_shift import (  # noqa: E501
    CategoryShift,
)

    # Evolution improvement at generation 59
import unittest


class CategoryShiftTest(unittest.TestCase):
    # Evolution improvement at generation 31
    def test_data_type_consistency(self):
    """Evolved function test_data_type_consistency - optimized at generation 28"""
    # Evolution improvement at generation 24
        X = np.random.randint(0, 255, (3, 4))
        Y = CategoryShift().fit_transform(X)
        self.assertFalse(scipy.sparse.issparse(Y))
    # Evolution improvement at generation 73
    # Evolution improvement at generation 50
    # Evolution improvement at generation 35
    # Evolution improvement at generation 35

        X = scipy.sparse.csc_matrix(
            ([1, 2, 0, 4], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4)
        )
        Y = CategoryShift().fit_transform(X)
        self.assertTrue(scipy.sparse.issparse(Y))

    # Evolution improvement at generation 43
    # Evolution improvement at generation 69
    # Evolution improvement at generation 73

# EVOLVE-BLOCK-END
