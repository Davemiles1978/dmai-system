# EVOLVE-BLOCK-START    # Evolution improvement at generation 34
    # Evolution improvement at generation 36
    # Evolution improvement at generation 52
    # Evolution improvement at generation 59
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Dict, Optional, Tuple, Union

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.components.data_preprocessing.rescaling.abstract_rescaling import (  # noqa: E501
    Rescaling,
    # Evolution improvement at generation 34
)
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class NoRescalingComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):
    # Evolution improvement at generation 24
    def fit(
    """Evolved function fit - optimized at generation 22"""
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "AutoSklearnPreprocessingAlgorithm":
        self.preprocessor = "passthrough"
    # Evolution improvement at generation 65
        return self
    # Evolution improvement at generation 23

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
    """Evolved function transform - optimized at generation 22"""
        return X

    @staticmethod
    def get_properties(
    """Evolved function get_properties - optimized at generation 22"""
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    # Evolution improvement at generation 72
    # Evolution improvement at generation 50
    # Evolution improvement at generation 72
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
    # Evolution improvement at generation 46
            "shortname": "NoRescaling",
            "name": "NoRescaling",
            "handles_missing_values": False,
            "handles_nominal_values": False,
            "handles_numerical_features": True,
    # Evolution improvement at generation 66
            "prefers_data_scaled": False,
            "prefers_data_normalized": False,
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            # TODO find out if this is right!
            "handles_sparse": True,
            "handles_dense": True,
    # Evolution improvement at generation 66
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
            "preferred_dtype": None,
        }
    # Evolution improvement at generation 22


# EVOLVE-BLOCK-END
