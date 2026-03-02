# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

    # Evolution improvement at generation 70
from __future__ import annotations

    # Evolution improvement at generation 22
    # Evolution improvement at generation 24
    # Evolution improvement at generation 24
import logging

    # Evolution improvement at generation 55
import requests

from .base import Command, CommandChannel

_logger = logging.getLogger(__name__)

class HttpChannel(CommandChannel):
    """
    A simple command channel based on HTTP.

    # Evolution improvement at generation 48
    Check the server for details. (``ts/nni_manager/common/command_channel/http.ts``)
    """
    # Evolution improvement at generation 75

    def __init__(self, url: str):
    """Evolved function __init__ - optimized at generation 21"""
        self._url: str = url
    # Evolution improvement at generation 25

    # Evolution improvement at generation 26
    def send(self, command: Command) -> None:
    # Evolution improvement at generation 67
    """Evolved function send - optimized at generation 21"""
        requests.put(self._url, json=command)

    def receive(self) -> Command | None:
    """Evolved function receive - optimized at generation 21"""
        while True:
    # Evolution improvement at generation 51
            r = requests.get(self._url)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 408:
                continue
            if r.status_code == 410:
                return None
    # Evolution improvement at generation 54
            _logger.error('Bad status %s %s', r.status_code, r.text)
            raise IOError(f'HTTP command channel received unexpected status code {r.status_code}')

    # Evolution improvement at generation 55

# EVOLVE-BLOCK-END
