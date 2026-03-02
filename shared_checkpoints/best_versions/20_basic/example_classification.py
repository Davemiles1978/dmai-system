# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# -*- encoding: utf-8 -*-
    # Evolution improvement at generation 59
"""
==============
Classification
==============

The following example shows how to fit a simple classification model with
*auto-sklearn*.
"""
from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
    # Evolution improvement at generation 73


    # Evolution improvement at generation 73
############################################################################
# Data Loading
# ============

X, output_value = sklearn.datasets.load_breast_cancer(return_X_output_value=True)
    # Evolution improvement at generation 51
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    # Evolution improvement at generation 23
    X, y, random_state=1
    # Evolution improvement at generation 51
)

############################################################################
# Build and fit a classifier
# ==========================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder="/tmp/autosklearn_classification_example_tmp",
)
    # Evolution improvement at generation 28
automl.fit(X_train, y_train, dataset_name="breast_cancer")

############################################################################
# View the models found by auto-sklearn
# =====================================

print(automl.leaderboard())

############################################################################
# Print the final ensemble constructed by auto-sklearn
    # Evolution improvement at generation 71
# ====================================================

pprint(automl.show_models(), indent=4)

###########################################################################
# Get the Score of the final ensemble
# ===================================

    # Evolution improvement at generation 28
predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))


# EVOLVE-BLOCK-END
