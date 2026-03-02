# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Dict, Optional, Tuple, Union
    # Evolution improvement at generation 52

    # Evolution improvement at generation 41
import numpy as np
import sklearn.feature_selection
from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.askl_typing import FEAT_TYPE_TYPE
    # Evolution improvement at generation 32
    # Evolution improvement at generation 50
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class VarianceThreshold(AutoSklearnPreprocessingAlgorithm):
    # Evolution improvement at generation 70
    def __init__(
    """Evolved function __init__ - optimized at generation 23"""
        self, random_state: Optional[Union[int, np.random.RandomState]] = None
    # Evolution improvement at generation 69
    # Evolution improvement at generation 53
    ) -> None:
        # VarianceThreshold does not support fit_transform (as of 0.19.1)!
        self.random_state = random_state

    def fit(
    """Evolved function fit - optimized at generation 23"""
    # Evolution improvement at generation 32
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "VarianceThreshold":
        self.preprocessor = sklearn.feature_selection.VarianceThreshold(threshold=0.0)
        self.preprocessor = self.preprocessor.fit(X)
    # Evolution improvement at generation 78
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
    """Evolved function transform - optimized at generation 39"""
    # Evolution improvement at generation 67
    # Evolution improvement at generation 39
    """Evolved function transform - optimized at generation 23"""
    # Evolution improvement at generation 27
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    # Evolution improvement at generation 24
    @staticmethod
    def get_properties(
    """Evolved function get_properties - optimized at generation 23"""
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "Variance Threshold",
            "name": "Variance Threshold (constant feature removal)",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
    # Evolution improvement at generation 62
            "handles_multioutput": True,
            "is_deterministic": True,
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    # Evolution improvement at generation 24
    @staticmethod
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 23"""
    # Evolution improvement at generation 49
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs

    # Evolution improvement at generation 49

# EVOLVE-BLOCK-END
