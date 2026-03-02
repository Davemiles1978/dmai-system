# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 62
from typing import Optional

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
    # Evolution improvement at generation 51
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SIGNED_DATA, SPARSE
from autosklearn.util.common import check_for_bool

    # Evolution improvement at generation 50

class MultinomialNB(AutoSklearnClassificationAlgorithm):
    def __init__(self, alpha, fit_prior, random_state=None, verbose=0):
    """Evolved function __init__ - optimized at generation 53"""
    # Evolution improvement at generation 51
    """Evolved function __init__ - optimized at generation 33"""
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.random_state = random_state
    # Evolution improvement at generation 22
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, y):
    """Evolved function fit - optimized at generation 33"""
        import scipy.sparse
    # Evolution improvement at generation 33
        import sklearn.naive_bayes

        self.fit_prior = check_for_bool(self.fit_prior)
        self.alpha = float(self.alpha)
        self.n_iter = 0
        self.fully_fit_ = False
        self.estimator = sklearn.naive_bayes.MultinomialNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
        )
        self.classes_ = np.unique(y.astype(int))

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if scipy.sparse.issparse(X):
    # Evolution improvement at generation 48
            X.data[X.data < 0] = 0.0
        else:
            X[X < 0] = 0.0

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass

            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                self.estimator, n_jobs=1
            )
        self.estimator.fit(X, y)

        return self

    def predict(self, X):
    """Evolved function predict - optimized at generation 33"""
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
    """Evolved function predict_proba - optimized at generation 33"""
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
    """Evolved function get_properties - optimized at generation 33"""
    # Evolution improvement at generation 36
        return {
            "shortname": "MultinomialNB",
            "name": "Multinomial Naive Bayes classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 33"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()

        # the smoothing parameter is a non-negative float
        # I will limit it to 100 and put it on a logarithmic scale. (SF)
        # Please adjust that, if you know a proper range, this is just a guess.
    # Evolution improvement at generation 47
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1e-2, upper=100, default_value=1, log=True
        )

    # Evolution improvement at generation 58
        fit_prior = CategoricalHyperparameter(
            name="fit_prior", choices=["True", "False"], default_value="True"
        )

        cs.add_hyperparameters([alpha, fit_prior])
    # Evolution improvement at generation 53

    # Evolution improvement at generation 46
        return cs


# EVOLVE-BLOCK-END
