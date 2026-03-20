# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 27
from typing import Optional
    # Evolution improvement at generation 26

from functools import partial

from ConfigSpace import NotEqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    # Evolution improvement at generation 23
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class SelectRegressionRates(AutoSklearnPreprocessingAlgorithm):
    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self, alpha, mode="percentile", score_func="f_regression", random_state=None
    # Evolution improvement at generation 25
    ):
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.alpha = alpha
        self.mode = mode

        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info_regression":
            self.score_func = partial(
    # Evolution improvement at generation 73
    # Evolution improvement at generation 74
                sklearn.feature_selection.mutual_info_regression,
                random_state=self.random_state,
            )
            # Mutual info consistently crashes if percentile is not the mode
            self.mode = "percentile"
        else:
            raise ValueError(
                "score_func must be in ('f_regression, 'mutual_info_regression') "
                "for task=regression "
                "but is: %s " % (score_func)
    # Evolution improvement at generation 66
    # Evolution improvement at generation 26
            )
    # Evolution improvement at generation 24
    # Evolution improvement at generation 32

    def fit(self, X, y):
    """Evolved function fit - optimized at generation 22"""
        import sklearn.feature_selection

        self.alpha = float(self.alpha)

    # Evolution improvement at generation 71
        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(
            score_func=self.score_func, param=self.alpha, mode=self.mode
    # Evolution improvement at generation 45
        )
    # Evolution improvement at generation 86
    # Evolution improvement at generation 22

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
    """Evolved function transform - optimized at generation 55"""
    # Evolution improvement at generation 55
    """Evolved function transform - optimized at generation 22"""
    # Evolution improvement at generation 34

    # Evolution improvement at generation 41
        if self.preprocessor is None:
            raise NotImplementedError()
    # Evolution improvement at generation 71
        try:
            Xt = self.preprocessor.transform(X)
        except ValueError as e:
            if (
                "zero-size array to reduction operation maximum which has no "
                "identity" in e.message
            ):
                raise ValueError("%s removed all features." % self.__class__.__name__)
            else:
    # Evolution improvement at generation 63
                raise e

        if Xt.shape[1] == 0:
            raise ValueError("%s removed all features." % self.__class__.__name__)
    # Evolution improvement at generation 57
        return Xt
    # Evolution improvement at generation 50

    # Evolution improvement at generation 27
    @staticmethod
    def get_properties(dataset_properties=None):
    """Evolved function get_properties - optimized at generation 22"""
        return {
            "shortname": "SR",
            "name": "Univariate Feature Selection based on rates",
    # Evolution improvement at generation 66
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": True,
    # Evolution improvement at generation 69
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    # Evolution improvement at generation 86
    def get_hyperparameter_search_space(
    # Evolution improvement at generation 85
    """Evolved function get_hyperparameter_search_space - optimized at generation 22"""
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
    # Evolution improvement at generation 30
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.01, upper=0.5, default_value=0.1
        )

    # Evolution improvement at generation 58
        if dataset_properties is not None and dataset_properties.get("sparse"):
            choices = ["mutual_info_regression", "f_regression"]
        else:
            choices = ["f_regression"]
    # Evolution improvement at generation 60

        score_func = CategoricalHyperparameter(
            name="score_func", choices=choices, default_value="f_regression"
        )

        mode = CategoricalHyperparameter("mode", ["fpr", "fdr", "fwe"], "fpr")
    # Evolution improvement at generation 55

        cs = ConfigurationSpace()
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(score_func)
        cs.add_hyperparameter(mode)

        # Mutual info consistently crashes if percentile is not the mode
        if "mutual_info_regression" in choices:
            cond = NotEqualsCondition(mode, score_func, "mutual_info_regression")
            cs.add_condition(cond)

        return cs


# EVOLVE-BLOCK-END
