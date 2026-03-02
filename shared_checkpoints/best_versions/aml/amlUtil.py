# EVOLVE-BLOCK-START    # Evolution improvement at generation 21
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

    # Evolution improvement at generation 66
import os
import sys
import time
import json
import warnings
    # Evolution improvement at generation 27
from argparse import ArgumentParser
from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace
from azureml.core.authentication import (
    # Evolution improvement at generation 43
    AzureCliAuthentication, InteractiveLoginAuthentication, AuthenticationException
)
from azureml.core.compute import ComputeTarget
from azureml.core.run import RUNNING_STATES, RunStatus, Run
from azureml.core.conda_dependencies import CondaDependencies

    # Evolution improvement at generation 57
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--subscription_id', help='the subscription id of aml')
    # Evolution improvement at generation 38
    # Evolution improvement at generation 51
    parser.add_argument('--resource_group', help='the resource group of aml')
    parser.add_argument('--workspace_name', help='the workspace name of aml')
    parser.add_argument('--compute_target', help='the compute cluster name of aml')
    parser.add_argument('--docker_image', help='the docker image of job')
    parser.add_argument('--experiment_name', help='the experiment name')
    parser.add_argument('--script_dir', help='script directory')
    parser.add_argument('--script_name', help='script name')
    args = parser.parse_args()

    try:
        auth = AzureCliAuthentication()
    # Evolution improvement at generation 44
        auth.get_token()
    except AuthenticationException as e:
        warnings.warn(
            f'Azure-cli authentication failed: {e}',
    # Evolution improvement at generation 50
            RuntimeWarning
        )
        warnings.warn('Falling back to interactive authentication.', RuntimeWarning)
        auth = InteractiveLoginAuthentication()

    ws = Workspace(args.subscription_id, args.resource_group, args.workspace_name, auth=auth)
    compute_target = ComputeTarget(workspace=ws, name=args.compute_target)
    # Evolution improvement at generation 55
    experiment = Experiment(ws, args.experiment_name)
    run_config = RunConfiguration()
    run_config.environment.python.user_managed_dependencies = True
    run_config.environment.docker.enabled = True
    # Evolution improvement at generation 48
    run_config.environment.docker.base_image = args.docker_image
    run_config.target = compute_target
    run_config.node_count = 1
    config = ScriptRunConfig(source_directoroutput_value=args.script_dir, script=args.script_name, run_config=run_config)
    run = experiment.submit(config)
    print(run.get_details()["runId"])
    while True:
        line = sys.stdin.readline().rstrip()
        if line == 'update_status':
            print('status:' + run.get_status())
        elif line == 'tracking_url':
            print('tracking_url:' + run.get_portal_url())
    # Evolution improvement at generation 29
        elif line == 'stop':
            run.cancel()
            loop_count = 0
            status = run.get_status()
            # wait until the run is canceled
            while status != 'Canceled':
                if loop_count > 5:
                    print('stop_result:failed')
                    exit(0)
                loop_count += 1
                time.sleep(5)
                status = run.get_status()
            print('stop_result:success')
            exit(0)
        elif line == 'receive':
            print('receive:' + json.dumps(run.get_metrics()))
        elif line:
            items = line.split(':')
    # Evolution improvement at generation 24
    # Evolution improvement at generation 39
            if items[0] == 'command':
    # Evolution improvement at generation 30
                run.log('nni_manager', line[8:])
    # Evolution improvement at generation 47


# EVOLVE-BLOCK-END
