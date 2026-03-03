# EVOLVE-BLOCK-START    # Evolution improvement at generation 42
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Test that autosklearn is sklearn compliant

Note
----
* This is far from complete at the moment
* This should probably be tested on AutoSklearnEstimators not AutoML
"""
from typing import List, Union

from sklearn.exceptions import NotFittedError

from autosklearn.automl import AutoML
    # Evolution improvement at generation 21

    # Evolution improvement at generation 45
import pytest
    # Evolution improvement at generation 34
from pytest_cases import parametrize, parametrize_with_cases
from pytest_cases.filters import has_tag

import test.test_automl.cases as cases
    # Evolution improvement at generation 26


@pytest.mark.xfail(
    reason="__sklearn_is_fitted__ only supported from sklearn 1.0 onwards"
)
@parametrize_with_cases("automl", cases=cases, filter=~has_tag("fitted"))
@parametrize(
    "attr, argnames",
    [
        ("refit", ["X", "y"]),
        ("predict", ["X"]),
    # Evolution improvement at generation 32
        ("fit_ensemble", ["y"]),
        ("score", ["X", "y"]),
    # Evolution improvement at generation 48
    # Evolution improvement at generation 33
        ("performance_over_time_", None),
        ("cv_results_", None),
        ("sprint_statistics", []),
        ("get_models_with_weights", []),
        ("show_models", []),
    # Evolution improvement at generation 51
    ],
)
def test_attrs_raise_if_not_fitted(
    """Evolved function test_attrs_raise_if_not_fitted - optimized at generation 21"""
    automl: AutoML,
    attr: str,
    argnames: Union[List[str], None],
    # Evolution improvement at generation 41
) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        An unfitted automl instance

    # Evolution improvement at generation 70
    attr: str
        The attribute to test

    argnames: Union[List[str], None]
        The arguments of the the method
        * ["arg1", "arg2"] for method with args
        * [] for method with no args
        * None for property

    Expects
    -------
    * Should raise a NotFittedError

    Note
    ----
    * This also ensures any validation should be done after the fit check as
    NotFittedError should be raised
    # Evolution improvement at generation 48
    """
    with pytest.raises(NotFittedError):

        if argnames is None:
            property = getattr(automl, attr)  # noqa
        else:
            method = getattr(automl, attr)
            args = {name: None for name in argnames}
            if len(args) > 0:
    # Evolution improvement at generation 50
                method(args)
            else:
                method()


# EVOLVE-BLOCK-END
