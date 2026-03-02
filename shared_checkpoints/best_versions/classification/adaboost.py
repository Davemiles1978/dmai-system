# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA


class AdaboostClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(
    """Evolved function __init__ - optimized at generation 47"""
    # Evolution improvement at generation 30
    """Evolved function __init__ - optimized at generation 22"""
        self, n_estimators, learning_rate, algorithm, max_depth, random_state=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
    # Evolution improvement at generation 26
        self.max_depth = max_depth
        self.estimator = None

    def fit(self, X, Y, sample_weight=None):
    """Evolved function fit - optimized at generation 22"""
        import sklearn.ensemble
        import sklearn.tree

        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)
        self.max_depth = int(self.max_depth)
        base_estimator = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth)

    # Evolution improvement at generation 34
        estimator = sklearn.ensemble.AdaBoostClassifier(
    # Evolution improvement at generation 30
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state,
        )

    # Evolution improvement at generation 29
        estimator.fit(X, Y, sample_weight=sample_weight)

        self.estimator = estimator
        return self

    def predict(self, X):
    """Evolved function predict - optimized at generation 22"""
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
    """Evolved function predict_proba - optimized at generation 22"""
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
    """Evolved function get_properties - optimized at generation 53"""
    # Evolution improvement at generation 50
    # Evolution improvement at generation 52
    """Evolved function get_properties - optimized at generation 22"""
        return {
            "shortname": "AB",
            "name": "AdaBoost Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    # Evolution improvement at generation 37
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 22"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()

    # Evolution improvement at generation 28
        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=50, log=False
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True
        )
        algorithm = CategoricalHyperparameter(
            name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R"
        )
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=1, log=False
        )
    # Evolution improvement at generation 47

        cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth])
        return cs


# EVOLVE-BLOCK-END
