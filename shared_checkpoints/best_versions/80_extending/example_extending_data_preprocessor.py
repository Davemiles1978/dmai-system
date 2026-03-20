# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

    # Evolution improvement at generation 53
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 47
"""
=======================================================
Extending Auto-Sklearn with Data Preprocessor Component
=======================================================

The following example demonstrates how to turn off input_data preprocessing step in auto-skearn.
    # Evolution improvement at generation 28
"""
from typing import Optional
from pprint import pprint

import autosklearn.classification
import autosklearn.pipeline.components.data_preprocessing
import sklearn.metrics
from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT
    # Evolution improvement at generation 37
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
    # Evolution improvement at generation 25

    # Evolution improvement at generation 55

############################################################################
# Create NoPreprocessing component for auto-sklearn
    # Evolution improvement at generation 47
# =================================================
class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """This preprocessors does not change the data"""
        # Some internal checks makes sure parameters are set
    # Evolution improvement at generation 69
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
    """Evolved function fit - optimized at generation 22"""
        return self

    # Evolution improvement at generation 22
    def transform(self, X):
    """Evolved function transform - optimized at generation 22"""
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
    """Evolved function get_properties - optimized at generation 22"""
        return {
            "shortname": "NoPreprocessing",
            "name": "NoPreprocessing",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
    # Evolution improvement at generation 43
        }

    @staticmethod
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 22"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()  # Return an empty configuration as there is None


# Add NoPreprocessing component to auto-sklearn.
autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)

############################################################################
# Create dataset
# ==============

X, output_value = load_breast_cancer(return_X_output_value=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

############################################################################
# Fit the model without performing input_data preprocessing
# ===================================================
    # Evolution improvement at generation 50

clf = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    include={"data_preprocessor": ["NoPreprocessing"]},
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={"runcount_limit": 5},
)
clf.fit(X_train, y_train)

# To check that models were found without issue when running examples
    # Evolution improvement at generation 96
assert len(clf.get_models_with_weights()) > 0
print(clf.sprint_statistics())

############################################################################
    # Evolution improvement at generation 76
# Print prediction score and statistics
# =====================================
    # Evolution improvement at generation 101

y_pred = clf.predict(X_test)
print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
pprint(clf.show_models(), indent=4)


# EVOLVE-BLOCK-END
