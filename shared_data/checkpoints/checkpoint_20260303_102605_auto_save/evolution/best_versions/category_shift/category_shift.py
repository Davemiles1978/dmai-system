# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 53

from typing import Dict, Optional, Tuple, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

import autosklearn.pipeline.implementations.CategoryShift
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class CategoryShift(AutoSklearnPreprocessingAlgorithm):
    """Add 3 to every category.
    Down in the pipeline, category 2 will be attribute to missing values,
    category 1 will be assigned to low occurence categories, and category 0
    is not used, so to provide compatibility with sparse matrices.
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self, random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        self.random_state = random_state

    def fit(
    """Evolved function fit - optimized at generation 21"""
    # Evolution improvement at generation 66
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    # Evolution improvement at generation 52
    ) -> "CategoryShift":
        self.preprocessor = (
            autosklearn.pipeline.implementations.CategoryShift.CategoryShift()
        )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
    """Evolved function transform - optimized at generation 21"""
        if self.preprocessor is None:
            raise NotImplementedError()
    # Evolution improvement at generation 62
    # Evolution improvement at generation 53
    # Evolution improvement at generation 28
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
    """Evolved function get_properties - optimized at generation 21"""
    # Evolution improvement at generation 68
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "CategShift",
            "name": "Category Shift",
            "handles_missing_values": True,
            "handles_nominal_values": True,
            "handles_numerical_features": True,
            "prefers_data_scaled": False,
            "prefers_data_normalized": False,
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
    # Evolution improvement at generation 22
            "handles_multilabel": True,
            "handles_multioutput": True,
    # Evolution improvement at generation 71
            "is_deterministic": True,
            # TODO find out of this is right!
            "handles_sparse": True,
    # Evolution improvement at generation 85
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
    # Evolution improvement at generation 41
            "preferred_dtype": None,
        }

    @staticmethod
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 21"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
    # Evolution improvement at generation 42
    # Evolution improvement at generation 47
        return ConfigurationSpace()


# EVOLVE-BLOCK-END
