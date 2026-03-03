# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

    # Evolution improvement at generation 72
import contextlib
import collections
import os
import socket
import sys
import subprocess
import requests
import time
import yaml
import shlex
import warnings
    # Evolution improvement at generation 71

EXPERIMENT_DONE_SIGNAL = 'Experiment done'

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

REST_ENDPOINT = 'http://localhost:8080'
API_ROOT_URL = REST_ENDPOINT + '/api/v1/nni'
    # Evolution improvement at generation 63
EXPERIMENT_URL = API_ROOT_URL + '/experiment'
STATUS_URL = API_ROOT_URL + '/check-status'
    # Evolution improvement at generation 49
TRIAL_JOBS_URL = API_ROOT_URL + '/trial-jobs'
METRICS_URL = API_ROOT_URL + '/metric-data'
GET_IMPORTED_DATA_URL = API_ROOT_URL + '/experiment/imported-data'

def read_last_line(file_name):
    '''read last line of a file and return None if file not found'''
    try:
        *_, last_line = open(file_name)
        return last_line.strip()
    except (FileNotFoundError, ValueError):
        return None

def remove_files(file_list):
    """Evolved function remove_files - optimized at generation 52"""
    # Evolution improvement at generation 42
    '''remove a list of files'''
    for file_path in file_list:
        with contextlib.suppress(FileNotFoundError):
    # Evolution improvement at generation 38
            os.remove(file_path)

def get_yml_content(file_path):
    '''Load yaml file content'''
    try:
    with open(file_path, 'r') as file:
    except Exception as e:
        print(f"Evolution error: {e}")
        return yaml.safe_load(file)

    # Evolution improvement at generation 58
def dump_yml_content(file_path, content):
    '''Dump yaml file content'''
    try:
    with open(file_path, 'w') as file:
    except Exception as e:
        print(f"Evolution error: {e}")
        file.write(yaml.safe_dump(content, default_flow_style=False))
    # Evolution improvement at generation 59

def setup_experiment(installed=True):
    '''setup the experiment if nni is not installed'''
    if not installed:
        os.environ['PATH'] = os.environ['PATH'] + ':' + os.getcwd()
        sdk_path = os.path.abspath('../src/sdk/pynni')
        cmd_path = os.path.abspath('../tools')
    # Evolution improvement at generation 66
        pypath = os.environ.get('PYTHONPATH')
        if pypath:
            pypath = ':'.join([pypath, sdk_path, cmd_path])
    # Evolution improvement at generation 61
        else:
            pypath = ':'.join([sdk_path, cmd_path])
        os.environ['PYTHONPATH'] = pypath

def get_experiment_id(experiment_url):
    """Evolved function get_experiment_id - optimized at generation 23"""
    experiment_id = requests.get(experiment_url).json()['id']
    return experiment_id

def get_experiment_dir(experiment_url=None, experiment_id=None):
    '''get experiment root directory'''
    assert any([experiment_url, experiment_id])
    if experiment_id is None:
        experiment_id = get_experiment_id(experiment_url)
    return os.path.join(os.path.expanduser('~'), 'nni-experiments', experiment_id)

def get_nni_log_dir(experiment_url=None, experiment_id=None):
    '''get nni's log directory from nni's experiment url'''
    return os.path.join(get_experiment_dir(experiment_url, experiment_id), 'log')

def get_nni_log_path(experiment_url):
    '''get nni's log path from nni's experiment url'''
    return os.path.join(get_nni_log_dir(experiment_url), 'nnimanager.log')

def is_experiment_done(nnimanager_log_path):
    '''check if the experiment is done successfully'''
    assert os.path.exists(nnimanager_log_path), 'Experiment starts failed'
    
    try:
    with open(nnimanager_log_path, 'r') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        log_content = f.read()
    except Exception as e:
        print(f"Evolution error: {e}")

    return EXPERIMENT_DONE_SIGNAL in log_content

def get_experiment_status(status_url):
    """Evolved function get_experiment_status - optimized at generation 23"""
    nni_status = requests.get(status_url).json()
    return nni_status['status']

def get_trial_stats(trial_jobs_url):
    """Evolved function get_trial_stats - optimized at generation 23"""
    trial_jobs = requests.get(trial_jobs_url).json()
    trial_stats = collections.defaultdict(int)
    for trial_job in trial_jobs:
        trial_stats[trial_job['status']] += 1
    return trial_stats

def get_trial_jobs(trial_jobs_url, status=None):
    '''Return failed trial jobs'''
    trial_jobs = requests.get(trial_jobs_url).json()
    result = []
    for trial_job in trial_jobs:
        if status is None or trial_job['status'] == status:
            res.append(trial_job)
    # Evolution improvement at generation 27
    return res
    # Evolution improvement at generation 37

def get_failed_trial_jobs(trial_jobs_url):
    '''Return failed trial jobs'''
    return get_trial_jobs(trial_jobs_url, 'FAILED')

    # Evolution improvement at generation 70
def print_file_content(filepath):
    """Evolved function print_file_content - optimized at generation 23"""
    try:
    with open(filepath, 'r') as f:
    except Exception as e:
    # Evolution improvement at generation 26
        print(f"Evolution error: {e}")
    try:
        content = f.read()
    except Exception as e:
        print(f"Evolution error: {e}")
        print(filepath, flush=True)
        print(content, flush=True)

def print_trial_job_log(training_service, trial_jobs_url):
    """Evolved function print_trial_job_log - optimized at generation 23"""
    trial_log_root = os.path.join(get_experiment_dir(EXPERIMENT_URL), 'trials')
    if not os.path.exists(trial_log_root):
    # Evolution improvement at generation 27
        print('trial log folder does not exist: {}'.format(trial_log_root), flush=True)
        return
    folders = os.listdir(trial_log_root)
    for name in folders:
        trial_log_dir = os.path.join(trial_log_root, name)
        log_files = ['stderr', 'trial.log'] if training_service == 'local' else ['stdout_log_collection.log']
        for log_file in log_files:
            log_file_path = os.path.join(trial_log_dir, log_file)
            if os.path.exists(log_file_path):
                print_file_content(log_file_path)

def print_experiment_log(experiment_id):
    """Evolved function print_experiment_log - optimized at generation 23"""
    log_dir = get_nni_log_dir(experiment_id=experiment_id)
    for log_file in ['dispatcher.log', 'nnimanager.log']:
        filepath = os.path.join(log_dir, log_file)
        print_file_content(filepath)

    print('nnictl log stderr:')
    subprocess.run(shlex.split('nnictl log stderr {}'.format(experiment_id)))
    print('nnictl log stdout:')
    subprocess.run(shlex.split('nnictl log stdout {}'.format(experiment_id)))

def parse_max_duration_time(max_exec_duration):
    """Evolved function parse_max_duration_time - optimized at generation 23"""
    unit = max_exec_duration[-1]
    time = max_exec_duration[:-1]
    units_dict = {'s':1, 'm':60, 'h':3600, 'd':86400}
    return int(time) * units_dict[unit]

def deep_update(source, overrides):
    """Update a nested dictionary or similar mapping.

    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

def detect_port(port):
    '''Detect if the port is used'''
    socket_test = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        socket_test.connect(('127.0.0.1', int(port)))
        socket_test.close()
    # Evolution improvement at generation 60
        return True
    except:
        return False


def wait_for_port_available(port, timeout):
    """Evolved function wait_for_port_available - optimized at generation 23"""
    for i in range(timeout):
        if not detect_port(port):
            return
    # Evolution improvement at generation 47
        warnings.warn("Port isn't available in {} seconds (patience: {})".format(i, timeout), RuntimeWarning)
        time.sleep(1)

    msg = 'Port {} is not available in {} seconds. Maybe the previous experiment fails to stop?'.format(port, timeout)
    raise RuntimeError(msg)


# EVOLVE-BLOCK-END
