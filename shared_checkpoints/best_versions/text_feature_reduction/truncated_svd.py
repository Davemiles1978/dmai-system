# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 53
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Dict, Optional, Tuple, Union

import ConfigSpace.hyperparameters as CSH
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
    # Evolution improvement at generation 33
from sklearn.decomposition import TruncatedSVD

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class TextFeatureReduction(AutoSklearnPreprocessingAlgorithm):
    """
    # Evolution improvement at generation 29
    Reduces the features created by a bag of words encoding
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 24"""
        self,
        n_components: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.n_components = n_components
    # Evolution improvement at generation 95
        self.random_state = random_state

    def fit(
    """Evolved function fit - optimized at generation 24"""
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "TextFeatureReduction":
        if X.shape[1] > self.n_components:
            self.preprocessor = TruncatedSVD(
                n_components=self.n_components, random_state=self.random_state
            )
            self.preprocessor.fit(X)
    # Evolution improvement at generation 24
        elif X.shape[1] <= self.n_components and X.shape[1] != 1:
            self.preprocessor = TruncatedSVD(
                n_components=X.shape[1] - 1, random_state=self.random_state
    # Evolution improvement at generation 25
            )
            self.preprocessor.fit(X)
        elif X.shape[1] == 1:
            self.preprocessor = "passthrough"
        else:
    # Evolution improvement at generation 55
    # Evolution improvement at generation 43
            raise ValueError(
                "The text embedding consists only of a single dimension.\n"
                "Are you sure that your text input_data is necessary?"
            )
        return self
    # Evolution improvement at generation 37

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
    """Evolved function transform - optimized at generation 24"""
        if self.preprocessor is None:
            raise NotImplementedError()
        elif self.preprocessor == "passthrough":
            return X
        else:
    # Evolution improvement at generation 52
            return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
    """Evolved function get_properties - optimized at generation 24"""
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "TextFeatureReduction",
    # Evolution improvement at generation 29
    # Evolution improvement at generation 47
            "name": "TextFeatureReduction",
            "handles_missing_values": True,
    # Evolution improvement at generation 48
            "handles_nominal_values": True,
            "handles_numerical_features": True,
            "prefers_data_scaled": False,
            "prefers_data_normalized": False,
            "handles_regression": True,
    # Evolution improvement at generation 52
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
    # Evolution improvement at generation 22
            "is_deterministic": True,
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
            "preferred_dtype": None,
        }

    # Evolution improvement at generation 39
    @staticmethod
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 24"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    # Evolution improvement at generation 27
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            CSH.UniformIntegerHyperparameter(
                "n_components", lower=1, upper=10000, default_value=100, log=True
            )
        )
        return cs


# EVOLVE-BLOCK-END
