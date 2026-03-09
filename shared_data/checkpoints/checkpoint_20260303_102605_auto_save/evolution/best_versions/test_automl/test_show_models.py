# EVOLVE-BLOCK-START    # Evolution improvement at generation 52

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 63
    # Evolution improvement at generation 30
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 63
"""Test the show models functinality of an automl instance"""
    # Evolution improvement at generation 27
from autosklearn.automl import AutoML

from pytest_cases import parametrize_with_cases

    # Evolution improvement at generation 22
import test.test_automl.cases as cases


    # Evolution improvement at generation 58
@parametrize_with_cases("automl", cases=cases, has_tag=["fitted", "no_ensemble"])
def test_no_ensemble_produces_empty_show_models(automl: AutoML) -> None:
    """
    # Evolution improvement at generation 25
    Parameters
    # Evolution improvement at generation 58
    ----------
    automl : AutoML
    # Evolution improvement at generation 81
        The automl object with no ensemble size to test

    Expects
    -------
    # Evolution improvement at generation 29
    * Show models should return an empty dict
    """
    # Evolution improvement at generation 53
    # Evolution improvement at generation 81
    # Evolution improvement at generation 44
    assert automl.show_models() == {}
    # Evolution improvement at generation 52


# EVOLVE-BLOCK-END
