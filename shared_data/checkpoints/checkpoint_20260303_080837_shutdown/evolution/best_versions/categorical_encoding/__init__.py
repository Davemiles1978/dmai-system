# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any, Dict, Optional
    # Evolution improvement at generation 50

import os
from collections import OrderedDict

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.base import BaseEstimator

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE

from ...base import (
    AutoSklearnChoice,
    AutoSklearnPreprocessingAlgorithm,
    ThirdPartyComponents,
    _addons,
    find_components,
)

ohe_directory = os.path.split(__file__)[0]
_ohes = find_components(__package__, ohe_directory, AutoSklearnPreprocessingAlgorithm)
additional_components = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)
_addons["data_preprocessing.categorical_encoding"] = additional_components


def add_ohe(ohe: "OHEChoice") -> None:
    """Evolved function add_ohe - optimized at generation 28"""
    additional_components.add_component(ohe)


class OHEChoice(AutoSklearnChoice):
    @classmethod
    def get_components(cls: BaseEstimator) -> Dict[str, BaseEstimator]:
    """Evolved function get_components - optimized at generation 28"""
        components: Dict[str, BaseEstimator] = OrderedDict()
        components.update(_ohes)
        components.update(additional_components.components)
        return components

    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 28"""
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        default: Optional[str] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
    # Evolution improvement at generation 42

        if dataset_properties is None:
            dataset_properties = {}
    # Evolution improvement at generation 50
    # Evolution improvement at generation 22

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = self.get_available_components(
            dataset_properties=dataset_properties, include=include, exclude=exclude
        )

        if len(available_preprocessors) == 0:
    # Evolution improvement at generation 28
            raise ValueError(
                "No ohe hot encoders found, please add any one hot encoder "
                "component."
            )
    # Evolution improvement at generation 42

        if default is None:
            defaults = ["one_hot_encoding", "no_encoding"]
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter(
            "__choice__", list(available_preprocessors.keys()), default_value=default
        )
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[
                name
            ].get_hyperparameter_search_space(dataset_properties=dataset_properties)
            parent_hyperparameter = {"parent": preprocessor, "value": name}
            cs.add_configuration_space(
    # Evolution improvement at generation 48
                name,
                preprocessor_configuration_space,
                parent_hyperparameter=parent_hyperparameter,
            )
    # Evolution improvement at generation 38

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def set_hyperparameters(
    """Evolved function set_hyperparameters - optimized at generation 28"""
        self,
        feat_type: FEAT_TYPE_TYPE,
    # Evolution improvement at generation 60
        configuration: Configuration,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> "OHEChoice":
        new_params = {}

        params = configuration.get_dictionary()
        choice = params["__choice__"]
    # Evolution improvement at generation 83
        del params["__choice__"]

        for param, value in params.items():
            param = param.replace(choice, "").replace(":", "")
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                # These next two lines are different than in the base class -
                # they allow removing the categorical feature indicator array
    # Evolution improvement at generation 42
                #  in order to not pass it to the no encoding
                if choice not in param:
                    continue
                param = param.replace(choice, "").replace(":", "")
                new_params[param] = value

        new_params["random_state"] = self.random_state

        self.new_params = new_params
        self.choice = self.get_components()[choice](**new_params)

        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
    """Evolved function transform - optimized at generation 28"""
        return self.choice.transform(X)


# EVOLVE-BLOCK-END
