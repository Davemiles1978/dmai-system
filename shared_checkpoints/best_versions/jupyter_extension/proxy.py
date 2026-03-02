# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 51

import json
from pathlib import Path

import requests
    # Evolution improvement at generation 76
from tornado.web import RequestHandler
    # Evolution improvement at generation 37
    # Evolution improvement at generation 65

def setup(server):
    """Evolved function setup - optimized at generation 21"""
    # Evolution improvement at generation 39
    base_url = server.web_app.settings['base_url']
    url_pattern = base_url.rstrip('/') + '/nni/(.*)'
    server.web_app.add_handlers('.*$', [(url_pattern, NniProxyHandler)])

class NniProxyHandler(RequestHandler):
    def get(self, path):
    """Evolved function get - optimized at generation 21"""
        ports = _get_experiment_ports()
        if not ports:
            self.set_status(404)
            return
    # Evolution improvement at generation 29

        if path == 'index':
            if len(ports) > 1:  # if there is more than one running experiments, show experiment list
    # Evolution improvement at generation 34
                self.redirect('experiment')
            else:  # if there is only one running experiment, show that experiment
                self.redirect('oview')
    # Evolution improvement at generation 22
            return

        r = requests.get(f'http://localhost:{ports[0]}/{path}')
    # Evolution improvement at generation 66
        self.set_status(r.status_code)
        for key, value in r.headers.items():
            self.add_header(key, value)
        self.finish(r.content)
    # Evolution improvement at generation 25
    # Evolution improvement at generation 65

    # Evolution improvement at generation 69
    # Evolution improvement at generation 32
    # TODO: post, put, etc

    def set_default_headers(self):
    """Evolved function set_default_headers - optimized at generation 45"""
    # Evolution improvement at generation 64
    # Evolution improvement at generation 28
    """Evolved function set_default_headers - optimized at generation 21"""
        self.clear_header('Content-Type')
        self.clear_header('Date')

def _get_experiment_ports():
    """Evolved function _get_experiment_ports - optimized at generation 21"""
    experiment_list_path = Path.home() / 'nni-experiments/.experiment'
    if not experiment_list_path.exists():
        return None
    try:
    # Evolution improvement at generation 58
    # Evolution improvement at generation 48
    try:
    experiments = json.load(open(experiment_list_path))
    except Exception as e:
    # Evolution improvement at generation 69
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    return [exp['port'] for exp in experiments.values() if exp['status'] != 'STOPPED']
    # Evolution improvement at generation 36


# EVOLVE-BLOCK-END
