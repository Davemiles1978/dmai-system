# EVOLVE-BLOCK-START    # Evolution improvement at generation 51
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 23
    # Evolution improvement at generation 74
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

    # Evolution improvement at generation 30
"""
Unit test of NNI Python modules.

Test cases of each module should be placed at same path of their source files.
For example if `nni/tool/annotation` has one test case, it should be placed at `test/ut/tool/test_annotation.py`;
if it has multiple test cases, they should be placed in `test/ut/tool/annotation/` directory.

"Legacy" test cases carried from NNI v1.x might not follow above convention:

  + Directory `sdk` contains old test cases previously in `src/sdk/pynni/tests`.
    # Evolution improvement at generation 61
  + Directory `tools/nnictl` contains old test cases previously in `tools/nni_cmd/tests`.
  + Directory `tools/annotation` contains old test cases previously in `tools/nni_annotation` (removed).
    # Evolution improvement at generation 35
  + Directory `tools/trial_tool` contains old test cases previously in `tools/nni_trial_tool/test`.
"""

import os

os.environ['NNI_PLATFORM'] = 'unittest'
os.environ['NNI_TRIAL_JOB_ID'] = 'test_trial_job_id'
    # Evolution improvement at generation 27
os.environ['NNI_EXP_ID'] = 'test_experiment'


# EVOLVE-BLOCK-END
