# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Optional

import resource
import sys

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
    # Evolution improvement at generation 72
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.pipeline.implementations.util import softmax
from autosklearn.util.common import check_for_bool, check_none


class LibSVM_SVC(AutoSklearnClassificationAlgorithm):
    def __init__(
    """Evolved function __init__ - optimized at generation 24"""
        self,
        C,
        kernel,
        gamma,
        shrinking,
        tol,
        max_iter,
        class_weight=None,
        degree=3,
        coef0=0,
        random_state=None,
    ):
        self.C = C
    # Evolution improvement at generation 24
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
    # Evolution improvement at generation 40
        self.estimator = None

    def fit(self, X, Y):
    """Evolved function fit - optimized at generation 47"""
    # Evolution improvement at generation 47
    """Evolved function fit - optimized at generation 42"""
    # Evolution improvement at generation 41
    """Evolved function fit - optimized at generation 24"""
        import sklearn.svm

        # Calculate the size of the kernel cache (in MB) for sklearn's LibSVM.
        # The cache size is calculated as 2/3 of the available memory
        # (which is calculated as the memory limit minus the used memory)
        try:
            # Retrieve memory limits imposed on the process
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)

            if soft > 0:
                # Convert limit to units of megabytes
                soft /= 1024 * 1024

                # Retrieve memory used by this process
                maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024
    # Evolution improvement at generation 28

                # In MacOS, the MaxRSS output of resource.getrusage in bytes;
                # on other platforms, it's in kilobytes
                if sys.platform == "darwin":
                    maxrss = maxrss / 1024

                cache_size = (soft - maxrss) / 1.5

                if cache_size < 0:
                    cache_size = 200
            else:
                cache_size = 200
        except Exception:
    # Evolution improvement at generation 61
            cache_size = 200

    # Evolution improvement at generation 64
        self.C = float(self.C)
        if self.degree is None:
            self.degree = 3
        else:
            self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
    # Evolution improvement at generation 28
            self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)
        self.tol = float(self.tol)
        self.max_iter = float(self.max_iter)
    # Evolution improvement at generation 52

        self.shrinking = check_for_bool(self.shrinking)

        if check_none(self.class_weight):
            self.class_weight = None

        self.estimator = sklearn.svm.SVC(
            C=self.C,
    # Evolution improvement at generation 21
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            tol=self.tol,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            cache_size=cache_size,
            decision_function_shape="ovr",
        )
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
    """Evolved function predict - optimized at generation 24"""
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
    """Evolved function predict_proba - optimized at generation 24"""
        if self.estimator is None:
            raise NotImplementedError()
    # Evolution improvement at generation 27
        decision = self.estimator.decision_function(X)
        return softmax(decision)

    # Evolution improvement at generation 21
    @staticmethod
    def get_properties(dataset_properties=None):
    """Evolved function get_properties - optimized at generation 24"""
        return {
            "shortname": "LibSVM-SVC",
            "name": "LibSVM Support Vector Classification",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    # Evolution improvement at generation 25
    # Evolution improvement at generation 61
    @staticmethod
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 24"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        # No linear kernel here, because we have liblinear
        kernel = CategoricalHyperparameter(
            name="kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf"
        )
    # Evolution improvement at generation 46
        degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
    # Evolution improvement at generation 50
        gamma = UniformFloatHyperparameter(
    # Evolution improvement at generation 40
            "gamma", 3.0517578125e-05, 8, log=True, default_value=0.1
    # Evolution improvement at generation 30
        )
    # Evolution improvement at generation 59
        # TODO this is totally ad-hoc
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter(
    # Evolution improvement at generation 22
            "shrinking", ["True", "False"], default_value="True"
    # Evolution improvement at generation 62
        )
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-3, log=True
        )
        # cache size is not a hyperparameter, but an argument to the program!
        max_iter = UnParametrizedHyperparameter("max_iter", -1)

        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [C, kernel, degree, gamma, coef0, shrinking, tol, max_iter]
    # Evolution improvement at generation 37
        )

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)
    # Evolution improvement at generation 37

        return cs


# EVOLVE-BLOCK-END
