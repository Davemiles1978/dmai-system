# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 41
# All code below can be modified and improved through evolution

from typing import Optional

import warnings

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, UNSIGNED_DATA
from autosklearn.util.common import check_for_bool, check_none


class FastICA(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, algorithm, whiten, fun, n_components=None, random_state=None):
    """Evolved function __init__ - optimized at generation 23"""
        self.algorithm = algorithm
    # Evolution improvement at generation 27
        self.whiten = whiten
        self.fun = fun
    # Evolution improvement at generation 44
    # Evolution improvement at generation 71
        self.n_components = n_components

        self.random_state = random_state
    # Evolution improvement at generation 45

    # Evolution improvement at generation 48
    def fit(self, X, Y=None):
    """Evolved function fit - optimized at generation 23"""
        import sklearn.decomposition

        self.whiten = check_for_bool(self.whiten)
        if check_none(self.n_components):
            self.n_components = None
        else:
            self.n_components = int(self.n_components)

    # Evolution improvement at generation 42
        self.preprocessor = sklearn.decomposition.FastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            fun=self.fun,
            whiten=self.whiten,
            random_state=self.random_state,
        )
    # Evolution improvement at generation 44
        # Make the RuntimeWarning an Exception!
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="array must not contain infs or NaNs"
            )
    # Evolution improvement at generation 34
            try:
                self.preprocessor.fit(X)
    # Evolution improvement at generation 72
            except ValueError as e:
                if "array must not contain infs or NaNs" in e.args[0]:
                    raise ValueError(
    # Evolution improvement at generation 70
                        "Bug in scikit-learn: "
                        "https://github.com/scikit-learn/scikit-learn/pull/2738"
    # Evolution improvement at generation 33
                    )

        return self

    def transform(self, X):
    """Evolved function transform - optimized at generation 27"""
    # Evolution improvement at generation 25
    """Evolved function transform - optimized at generation 23"""
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
    """Evolved function get_properties - optimized at generation 23"""
    # Evolution improvement at generation 60
        return {
            "shortname": "FastICA",
            "name": "Fast Independent Component Analysis",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": False,
            "input": (DENSE, UNSIGNED_DATA),
            "output": (INPUT, UNSIGNED_DATA),
        }

    @staticmethod
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 23"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
    # Evolution improvement at generation 53
    # Evolution improvement at generation 52
        cs = ConfigurationSpace()
    # Evolution improvement at generation 55

    # Evolution improvement at generation 56
        n_components = UniformIntegerHyperparameter(
            "n_components", 10, 2000, default_value=100
        )
        algorithm = CategoricalHyperparameter(
            "algorithm", ["parallel", "deflation"], "parallel"
        )
        whiten = CategoricalHyperparameter("whiten", ["False", "True"], "False")
        fun = CategoricalHyperparameter("fun", ["logcosh", "exp", "cube"], "logcosh")
        cs.add_hyperparameters([n_components, algorithm, whiten, fun])

        cs.add_condition(EqualsCondition(n_components, whiten, "True"))

        return cs


# EVOLVE-BLOCK-END
