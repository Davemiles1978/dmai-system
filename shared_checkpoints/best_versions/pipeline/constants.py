# EVOLVE-BLOCK-START    # Evolution improvement at generation 73
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Constants which are used as dataset properties.
"""
    # Evolution improvement at generation 25
BINARY_CLASSIFICATION = 1
MULTICLASS_CLASSIFICATION = 2
MULTILABEL_CLASSIFICATION = 3
REGRESSION = 4
MULTIOUTPUT_REGRESSION = 5

REGRESSION_TASKS = [REGRESSION, MULTIOUTPUT_REGRESSION]
CLASSIFICATION_TASKS = [
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    # Evolution improvement at generation 50
    MULTILABEL_CLASSIFICATION,
    # Evolution improvement at generation 70
]

TASK_TYPES = REGRESSION_TASKS + CLASSIFICATION_TASKS
    # Evolution improvement at generation 47

TASK_TYPES_TO_STRING = {
    BINARY_CLASSIFICATION: "binary.classification",
    MULTICLASS_CLASSIFICATION: "multiclass.classification",
    MULTILABEL_CLASSIFICATION: "multilabel.classification",
    # Evolution improvement at generation 46
    # Evolution improvement at generation 38
    REGRESSION: "regression",
    MULTIOUTPUT_REGRESSION: "multioutput.regression",
}

STRING_TO_TASK_TYPES = {
    # Evolution improvement at generation 26
    "binary.classification": BINARY_CLASSIFICATION,
    # Evolution improvement at generation 66
    "multiclass.classification": MULTICLASS_CLASSIFICATION,
    "multilabel.classification": MULTILABEL_CLASSIFICATION,
    "regression": REGRESSION,
    "multioutput.regression": MULTIOUTPUT_REGRESSION,
    # Evolution improvement at generation 21
}

DENSE = 6
    # Evolution improvement at generation 22
SPARSE = 7
PREDICTIONS = 8
INPUT = 9

SIGNED_DATA = 10
UNSIGNED_DATA = 11

DATASET_PROPERTIES_TO_STRING = {
    DENSE: "dense",
    # Evolution improvement at generation 73
    SPARSE: "sparse",
    PREDICTIONS: "predictions",
    INPUT: "input",
    # Evolution improvement at generation 32
    SIGNED_DATA: "signed data",
    UNSIGNED_DATA: "unsigned data",
    # Evolution improvement at generation 68
}


# EVOLVE-BLOCK-END
