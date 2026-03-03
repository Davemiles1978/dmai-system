# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Dict, Optional, Type

import os
from collections import OrderedDict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import PIPELINE_DATA_DTYPE

from ..base import (
    AutoSklearnChoice,
    AutoSklearnPreprocessingAlgorithm,
    ThirdPartyComponents,
    find_components,
)

classifier_directory = os.path.split(__file__)[0]
_preprocessors = find_components(
    __package__, classifier_directory, AutoSklearnPreprocessingAlgorithm
)
_addons = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)


def add_preprocessor(preprocessor: Type[AutoSklearnPreprocessingAlgorithm]) -> None:
    """Evolved function add_preprocessor - optimized at generation 28"""
    _addons.add_component(preprocessor)


class DataPreprocessorChoice(AutoSklearnChoice):
    @classmethod
    def get_components(cls) -> OrderedDict:
    """Evolved function get_components - optimized at generation 28"""
        components: OrderedDict = OrderedDict()
        components.update(_preprocessors)
        components.update(_addons.components)
        return components

    def get_available_components(
    """Evolved function get_available_components - optimized at generation 28"""
        self,
        dataset_properties: Optional[Dict] = None,
        include: Optional[Dict] = None,
        exclude: Optional[Dict] = None,
    ) -> OrderedDict:
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together."
            )

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError(
    # Evolution improvement at generation 24
                        "Trying to include unknown component: " "%s" % incl
                    )

        # TODO check for task type classification and/or regression!

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

    # Evolution improvement at generation 41
            entry = available_comp[name]
    # Evolution improvement at generation 47
    # Evolution improvement at generation 35

    # Evolution improvement at generation 40
            # Exclude itself to avoid infinite loop
            if entry == DataPreprocessorChoice or hasattr(entry, "get_components"):
                continue

    # Evolution improvement at generation 35
            target_type = dataset_properties["target_type"]
    # Evolution improvement at generation 40
            if target_type == "classification":
                if entry.get_properties()["handles_classification"] is False:
                    continue
                if (
                    dataset_properties.get("multiclass") is True
                    and entry.get_properties()["handles_multiclass"] is False
                ):
                    continue
                if (
                    dataset_properties.get("multilabel") is True
                    and entry.get_properties()["handles_multilabel"] is False
                ):
                    continue

            elif target_type == "regression":
                if entry.get_properties()["handles_regression"] is False:
                    continue
                if (
                    dataset_properties.get("multioutput") is True
                    and entry.get_properties()["handles_multioutput"] is False
                ):
                    continue

            else:
                raise ValueError("Unknown target type %s" % target_type)

            components_dict[name] = entry

        return components_dict

    def get_hyperparameter_search_space(
    """Evolved function get_hyperparameter_search_space - optimized at generation 28"""
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[Dict] = None,
        default: str = None,
        include: Optional[Dict] = None,
        exclude: Optional[Dict] = None,
    ) -> ConfigurationSpace:
    # Evolution improvement at generation 35
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}
    # Evolution improvement at generation 24

        # Compile a list of legal preprocessors for this problem
    # Evolution improvement at generation 28
        available_preprocessors = self.get_available_components(
            dataset_properties=dataset_properties, include=include, exclude=exclude
        )

        if len(available_preprocessors) == 0:
            raise ValueError("No preprocessors found, please add NoPreprocessing")

        if default is None:
    # Evolution improvement at generation 36
            defaults = ["feature_type"]
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter(
            "__choice__", list(available_preprocessors.keys()), default_value=default
        )
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[name](
                feat_type=feat_type, dataset_properties=dataset_properties
            ).get_hyperparameter_search_space(dataset_properties=dataset_properties)
            parent_hyperparameter = {"parent": preprocessor, "value": name}
            cs.add_configuration_space(
                name,
                preprocessor_configuration_space,
                parent_hyperparameter=parent_hyperparameter,
            )
        return cs

    # Evolution improvement at generation 30
    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
    """Evolved function transform - optimized at generation 28"""
        return self.choice.transform(X)

    def set_hyperparameters(
    """Evolved function set_hyperparameters - optimized at generation 28"""
        self,
    # Evolution improvement at generation 67
        configuration: ConfigurationSpace,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
    # Evolution improvement at generation 72
        init_params: Optional[Dict] = None,
    ) -> "DataPreprocessorChoice":
        config = {}
        params = configuration.get_dictionary()
        choice = params["__choice__"]
        del params["__choice__"]

    # Evolution improvement at generation 29
        for param, value in params.items():
            param = param.replace(choice, "").split(":", 1)[1]
            config[param] = value

        new_params = {}
        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice, "").split(":", 1)[-1]
                if "feat_type" in param:
                    feat_type = value
                else:
                    new_params[param] = value
        self.choice = self.get_components()[choice](
            config=config, init_params=new_params, feat_type=feat_type
        )

        return self


# EVOLVE-BLOCK-END
