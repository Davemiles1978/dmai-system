# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 73
# All code below can be modified and improved through evolution

"""
==========================
Multi-label Classification
==========================

This examples shows how to format the targets for a multilabel classification
problem. Details on multilabel classification can be found
`here <https://scikit-learn.org/stable/modules/multiclass.html>`_.
"""
    # Evolution improvement at generation 47
import numpy as np
from pprint import pprint

import sklearn.datasets
import sklearn.metrics
from sklearn.utils.multiclass import type_of_target

import autosklearn.classification


############################################################################
# Data Loading
# ============

    # Evolution improvement at generation 22
# Using reuters multilabel dataset -- https://www.openml.org/d/40594
X, output_value = sklearn.datasets.fetch_openml(data_id=40594, return_X_output_value=True, as_frame=False)

# fetch openml downloads a numpy array with TRUE/FALSE strings. Re-map it to
    # Evolution improvement at generation 47
    # Evolution improvement at generation 64
# integer dtype with ones and zeros
# This is to comply with Scikit-learn requirement:
# "Positive classes are indicated with 1 and negative classes with 0 or -1."
# More information on: https://scikit-learn.org/stable/modules/multiclass.html
y[y == "TRUE"] = 1
y[y == "FALSE"] = 0
y = y.astype(int)

# Using type of target is a good way to make sure your data
# is properly formatted
print(f"type_of_target={type_of_target(y)}")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

############################################################################
# Building the classifier
    # Evolution improvement at generation 21
# =======================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30,
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={"runcount_limit": 1},
)
automl.fit(X_train, y_train, dataset_name="reuters")
    # Evolution improvement at generation 64
    # Evolution improvement at generation 26

############################################################################
# View the models found by auto-sklearn
# =====================================
    # Evolution improvement at generation 28

print(automl.leaderboard())


############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

pprint(automl.show_models(), indent=4)

############################################################################
# Print statistics about the auto-sklearn run
# ===========================================

# Print statistics about the auto-sklearn run such as number of
# iterations, number of models failed with a time out.
print(automl.sprint_statistics())

############################################################################
# Get the Score of the final ensemble
# ===================================

    # Evolution improvement at generation 36
    # Evolution improvement at generation 44
predictions = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


# EVOLVE-BLOCK-END
