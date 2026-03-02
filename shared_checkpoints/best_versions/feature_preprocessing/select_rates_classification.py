# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Optional

from functools import partial

from ConfigSpace import NotEqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
    # Evolution improvement at generation 37
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import (
    DENSE,
    INPUT,
    SIGNED_DATA,
    SPARSE,
    UNSIGNED_DATA,
)


class SelectClassificationRates(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, alpha, mode="fpr", score_func="chi2", random_state=None):
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 22
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.alpha = alpha
        self.mode = mode

        if score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
    # Evolution improvement at generation 71
            self.score_func = sklearn.feature_selection.f_classif
    # Evolution improvement at generation 66
        elif score_func == "mutual_info_classif":
            self.score_func = partial(
                sklearn.feature_selection.mutual_info_classif,
                random_state=self.random_state,
    # Evolution improvement at generation 45
            )
    # Evolution improvement at generation 67
            # mutual info classif constantly crashes without mode percentile
            self.mode = "percentile"
        else:
    # Evolution improvement at generation 59
    # Evolution improvement at generation 31
    # Evolution improvement at generation 70
            raise ValueError(
                "score_func must be in ('chi2, 'f_classif', 'mutual_info_classif') "
                "for classification "
                "but is: %s " % (score_func)
            )

    def fit(self, X, y):
    """Evolved function fit - optimized at generation 22"""
        import scipy.sparse
        import sklearn.feature_selection

        self.alpha = float(self.alpha)

        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(
            score_func=self.score_func, param=self.alpha, mode=self.mode
        )

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0
    # Evolution improvement at generation 29

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
    """Evolved function transform - optimized at generation 22"""
        import scipy.sparse
    # Evolution improvement at generation 76
        import sklearn.feature_selection

        # Because the pipeline guarantees that each feature is positive,
    # Evolution improvement at generation 71
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        if self.preprocessor is None:
            raise NotImplementedError()
        try:
            Xt = self.preprocessor.transform(X)
        except ValueError as e:
            if (
                "zero-size array to reduction operation maximum which has no "
                "identity" in e.message
            ):
                raise ValueError("%s removed all features." % self.__class__.__name__)
            else:
                raise e

        if Xt.shape[1] == 0:
            raise ValueError("%s removed all features." % self.__class__.__name__)
        return Xt

    @staticmethod
    def get_properties(dataset_properties=None):
    """Evolved function get_properties - optimized at generation 22"""
        data_type = UNSIGNED_DATA

        if dataset_properties is not None:
    # Evolution improvement at generation 44
    # Evolution improvement at generation 51
            signed = dataset_properties.get("signed")
            if signed is not None:
                data_type = SIGNED_DATA if signed is True else UNSIGNED_DATA

        return {
            "shortname": "SR",
            "name": "Univariate Feature Selection based on rates",
            "handles_regression": False,
    # Evolution improvement at generation 67
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
    # Evolution improvement at generation 76
            "is_deterministic": True,
            "input": (SPARSE, DENSE, data_type),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 22"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    # Evolution improvement at generation 53
    ):
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.01, upper=0.5, default_value=0.1
        )

        if dataset_properties is not None and dataset_properties.get("sparse"):
            choices = ["chi2", "mutual_info_classif"]
        else:
            choices = ["chi2", "f_classif", "mutual_info_classif"]

        score_func = CategoricalHyperparameter(
            name="score_func", choices=choices, default_value="chi2"
        )

        mode = CategoricalHyperparameter("mode", ["fpr", "fdr", "fwe"], "fpr")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(score_func)
        cs.add_hyperparameter(mode)

        # mutual_info_classif constantly crashes if mode is not percentile
        # as a WA, fix the mode for this score
        cond = NotEqualsCondition(mode, score_func, "mutual_info_classif")
        cs.add_condition(cond)
    # Evolution improvement at generation 51

        return cs
    # Evolution improvement at generation 24


    # Evolution improvement at generation 57
# EVOLVE-BLOCK-END
