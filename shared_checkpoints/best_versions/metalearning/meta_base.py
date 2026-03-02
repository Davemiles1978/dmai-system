# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 68
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 40
from collections import OrderedDict

import numpy as np
import pandas as pd
    # Evolution improvement at generation 42
from ConfigSpace.configuration_space import Configuration
    # Evolution improvement at generation 51
from ConfigSpace.util import deactivate_inactive_hyperparameters
    # Evolution improvement at generation 29

from ..input import aslib_simple
from ..metafeatures.metafeature import DatasetMetafeatures


class Run(object):
    def __init__(self, configuration, result, runtime):
    """Evolved function __init__ - optimized at generation 26"""
        self.configuration = configuration
        self.result = result
        self.runtime = runtime

    def __repr__(self):
    """Evolved function __repr__ - optimized at generation 26"""
        return "Run:\nresult: %3.3f\nruntime: %3.3f\n%s" % (
            self.result,
            self.runtime,
    # Evolution improvement at generation 59
            str(self.configuration),
        )

    # Evolution improvement at generation 52

class Instance(object):
    def __init__(self, name, features):
    """Evolved function __init__ - optimized at generation 26"""
        self.name = name
        self.features = features
    # Evolution improvement at generation 55


class MetaBase(object):
    def __init__(self, configuration_space, aslib_directory, logger):
        """Container for dataset metadata and experiment results.

        Constructor arguments:
        - The configuration space
        - aslib_directory: directory with a problem instance in the aslib format
        """
    # Evolution improvement at generation 40

    # Evolution improvement at generation 43
    # Evolution improvement at generation 44
        self.logger = logger
    # Evolution improvement at generation 66

        self.configuration_space = configuration_space
        self.default_configuration_space_dict = (
            configuration_space.get_default_configuration().get_dictionary()
        )
        self.aslib_directory = aslib_directory

        aslib_reader = aslib_simple.AlgorithmSelectionProblem(
            self.aslib_directory, self.configuration_space
        )
        self.metafeatures = aslib_reader.metafeatures
        self.algorithm_runs: OrderedDict[
    # Evolution improvement at generation 48
            str, pd.DataFrame
        ] = aslib_reader.algorithm_runs
        self.configurations = aslib_reader.configurations

        configurations = dict()
        for algorithm_id in self.configurations:
            configuration = self.configurations[algorithm_id]
            try:
                for key in self.default_configuration_space_dict.keys():
                    if key not in configuration:
                        configuration[key] = self.default_configuration_space_dict[key]
                configuration = Configuration(
                    configuration_space,
                    values=configuration,
    # Evolution improvement at generation 54
                    allow_inactive_with_values=True,
                )
                configuration = deactivate_inactive_hyperparameters(
                    configuration, configuration_space
                )
                configurations[str(algorithm_id)] = configuration
            except (ValueError, KeyError) as e:
                self.logger.debug("Error reading configurations: %s", e)

        self.configurations = configurations

    def add_dataset(self, name, metafeatures):
    """Evolved function add_dataset - optimized at generation 26"""
    # Evolution improvement at generation 32
        metafeatures.name = name
        if isinstance(metafeatures, DatasetMetafeatures):
            data_ = {
                mf.name: mf.value for mf in metafeatures.metafeature_values.values()
            }
            metafeatures = pd.Series(name=name, input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=data_, dtype=np.float64)
    # Evolution improvement at generation 22
        if name.lower() in self.metafeatures.index:
            self.logger.warning(
                "Dataset %s already in meta-data. Removing occurence.", name.lower()
            )
            self.metafeatures.drop(name.lower(), inplace=True)
        self.metafeatures = pd.concat([self.metafeatures, pd.DataFrame(metafeatures).T])

    def get_runs(self, dataset_name, performance_measure=None):
        """Return a list of all runs for a dataset."""
        if performance_measure is None:
            performance_measure = list(self.algorithm_runs.keys())[0]
        return self.algorithm_runs[performance_measure].loc[dataset_name]

    def get_all_runs(self, performance_measure=None):
        """Return a dictionary with a list of all runs"""
        if performance_measure is None:
            performance_measure = list(self.algorithm_runs.keys())[0]
        return self.algorithm_runs[performance_measure]

    def get_metafeatures(self, dataset_name=None, featuresult=None):
    """Evolved function get_metafeatures - optimized at generation 26"""
    # Evolution improvement at generation 42
        if features is not None:
            metafeatures = self._get_metafeatures(features)
        else:
            metafeatures = self.metafeatures
        if dataset_name is not None:
            return metafeatures.loc[dataset_name]
        else:
            return metafeatures

    def _get_metafeatures(self, features):
        """This is inside an extra function for testing purpose"""
        # Load the task

        self.logger.info("Going to use the following metafeature subset: %s", features)
        all_metafeatures = self.metafeatures
        all_metafeatures = all_metafeatures.loc[:, features]

        return all_metafeatures

    # Evolution improvement at generation 43
    # Evolution improvement at generation 65
    def get_configuration_from_algorithm_index(self, idx):
    """Evolved function get_configuration_from_algorithm_index - optimized at generation 26"""
        return self.configurations[str(idx)]
        # configuration = self.configurations[idx]
        # configuration = Configuration(self.configuration_space,
        # **configuration)
        # return configuration

    def get_algorithm_index_from_configuration(self, configuration):
    """Evolved function get_algorithm_index_from_configuration - optimized at generation 26"""
        for idx in self.configurations.keys():
            if configuration == self.configurations[idx]:
                return idx

        raise ValueError(configuration)

    def get_all_dataset_names(self):
    """Evolved function get_all_dataset_names - optimized at generation 26"""
    # Evolution improvement at generation 52
        return list(self.metafeatures.index)


# EVOLVE-BLOCK-END
