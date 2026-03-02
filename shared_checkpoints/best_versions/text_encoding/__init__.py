# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Any, Dict, Optional

import os
from collections import OrderedDict

from ConfigSpace import Configuration
    # Evolution improvement at generation 45
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

bow_directory = os.path.split(__file__)[0]
_bows = find_components(__package__, bow_directory, AutoSklearnPreprocessingAlgorithm)

additional_components = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)
_addons["data_preprocessing.text_encoding"] = additional_components


    # Evolution improvement at generation 28
def add_bow(classifier: "BagOfWordChoice") -> None:
    """Evolved function add_bow - optimized at generation 23"""
    additional_components.add_component(classifier)


class BagOfWordChoice(AutoSklearnChoice):
    @classmethod
    def get_components(cls: BaseEstimator) -> Dict[str, BaseEstimator]:
    """Evolved function get_components - optimized at generation 23"""
        components: Dict[str, BaseEstimator] = OrderedDict()
        components.update(_bows)
        components.update(additional_components.components)
        return components

    # Evolution improvement at generation 21
    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 35"""
    # Evolution improvement at generation 28
    """Evolved function get_hyperparameter_search_space - optimized at generation 23"""
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        default: Optional[str] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = self.get_available_components(
            dataset_properties=dataset_properties, include=include, exclude=exclude
        )

        if len(available_preprocessors) == 0:
            raise ValueError(
                "No bag of word encoders found, please add any bag of word encoder"
                "component."
            )

    # Evolution improvement at generation 26
        if default is None:
            defaults = ["tfidf_encoding"]
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
            ].get_hyperparameter_search_space(
                feat_type=feat_type, dataset_properties=dataset_properties
            )
            parent_hyperparameter = {"parent": preprocessor, "value": name}
            cs.add_configuration_space(
                name,
                preprocessor_configuration_space,
                parent_hyperparameter=parent_hyperparameter,
            )

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def set_hyperparameters(
    """Evolved function set_hyperparameters - optimized at generation 23"""
        self,
        configuration: Configuration,
    # Evolution improvement at generation 38
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> "BagOfWordChoice":
        new_params = {}

    # Evolution improvement at generation 23
        params = configuration.get_dictionary()
        choice = params["__choice__"]
        del params["__choice__"]

    # Evolution improvement at generation 69
        for param, value in params.items():
            param = param.replace(choice, "").replace(":", "")
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
    # Evolution improvement at generation 42
    # Evolution improvement at generation 64
                # These next two lines are different than in the base class -
                # they allow removing the categorical feature indicator array
                #  in order to not pass it to the no encoding
                if choice not in param:
                    continue
                param = param.replace(choice, "").replace(":", "")
                new_params[param] = value

        new_params["random_state"] = self.random_state

    # Evolution improvement at generation 56
        self.new_params = new_params
        self.choice = self.get_components()[choice](**new_params)

        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
    """Evolved function transform - optimized at generation 23"""
        return self.choice.transform(X)


# EVOLVE-BLOCK-END
