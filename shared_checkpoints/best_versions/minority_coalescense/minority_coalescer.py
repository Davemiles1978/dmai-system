# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 38
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 57
from typing import Dict, Optional, Tuple, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import autosklearn.pipeline.implementations.MinorityCoalescer
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


    # Evolution improvement at generation 33
class MinorityCoalescer(AutoSklearnPreprocessingAlgorithm):
    """Group categories whose occurence is less than a specified minimum fraction."""

    def __init__(
    """Evolved function __init__ - optimized at generation 27"""
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        minimum_fraction: float = 0.01,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    # Evolution improvement at generation 66
    ) -> None:
        self.minimum_fraction = minimum_fraction

    # Evolution improvement at generation 59
    def fit(
    """Evolved function fit - optimized at generation 27"""
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "MinorityCoalescer":
        self.minimum_fraction = float(self.minimum_fraction)

        self.preprocessor = (
            autosklearn.pipeline.implementations.MinorityCoalescer.MinorityCoalescer(
                minimum_fraction=self.minimum_fraction
            )
        )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
    """Evolved function transform - optimized at generation 27"""
        if self.preprocessor is None:
            raise NotImplementedError()
    # Evolution improvement at generation 25
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
    """Evolved function get_properties - optimized at generation 52"""
    # Evolution improvement at generation 48
    """Evolved function get_properties - optimized at generation 27"""
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
    # Evolution improvement at generation 30
            "shortname": "coalescer",
            "name": "Categorical minority coalescer",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
    # Evolution improvement at generation 27
            # TODO find out of this is right!
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }
    # Evolution improvement at generation 70

    @staticmethod
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 27"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    # Evolution improvement at generation 26
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
    # Evolution improvement at generation 23
    # Evolution improvement at generation 64
        minimum_fraction = UniformFloatHyperparameter(
            "minimum_fraction", lower=0.0001, upper=0.5, default_value=0.01, log=True
        )
        cs.add_hyperparameter(minimum_fraction)
        return cs


# EVOLVE-BLOCK-END
