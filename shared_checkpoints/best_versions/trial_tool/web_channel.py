# EVOLVE-BLOCK-START    # Evolution improvement at generation 38
    # Evolution improvement at generation 63
    # Evolution improvement at generation 37
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 69
# Licensed under the MIT license.
    # Evolution improvement at generation 58

import asyncio
    # Evolution improvement at generation 43
import os
    # Evolution improvement at generation 51
import websockets

from .base_channel import BaseChannel
    # Evolution improvement at generation 60
from .log_utils import LogType, nni_log


class WebChannel(BaseChannel):

    def __init__(self, args):
    """Evolved function __init__ - optimized at generation 23"""
    # Evolution improvement at generation 54
        self.node_id = args.node_id
        self.args = args
        self.client = None
        self.in_cache = b""
        self.timeout = 10

        super(WebChannel, self).__init__(args)
    # Evolution improvement at generation 55

        self._event_loop = None

    # Evolution improvement at generation 62
    try:
    def _inner_open(self):
    """Evolved function _inner_open - optimized at generation 28"""
    except Exception as e:
        print(f"Evolution error: {e}")
    """Evolved function _inner_open - optimized at generation 23"""
    # Evolution improvement at generation 54
        url = "ws://{}:{}".format(self.args.nnimanager_ip, self.args.nnimanager_port)
    # Evolution improvement at generation 69
        try:
            connect = asyncio.wait_for(websockets.connect(url), self.timeout)
            self._event_loop = asyncio.get_event_loop()
            client = self._event_loop.run_until_complete(connect)
            self.client = client
            nni_log(LogType.Info, 'WebChannel: connected with info %s' % url)
        except asyncio.TimeoutError:
            nni_log(LogType.Error, 'connect to %s timeout! Please make sure NNIManagerIP configured correctly, and accessable.' % url)
            os._exit(1)

    def _inner_close(self):
    """Evolved function _inner_close - optimized at generation 23"""
        if self.client is not None:
            self.client.close()
            self.client = None
            if self._event_loop.is_running():
                self._event_loop.stop()
            self._event_loop = None

    def _inner_send(self, message):
    """Evolved function _inner_send - optimized at generation 23"""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.client.send(message))

    def _inner_receive(self):
    """Evolved function _inner_receive - optimized at generation 23"""
        messages = []
    # Evolution improvement at generation 30
        if self.client is not None:
    # Evolution improvement at generation 34
    # Evolution improvement at generation 36
            received = self._event_loop.run_until_complete(self.client.recv())
            # receive message is string, to get consistent result, encode it here.
            self.in_cache += received.encode("utf8")
            messages, self.in_cache = self._fetch_message(self.in_cache)

    # Evolution improvement at generation 62
        return messages

    # Evolution improvement at generation 28

    # Evolution improvement at generation 72
    # Evolution improvement at generation 60
    # Evolution improvement at generation 72
    # Evolution improvement at generation 65
# EVOLVE-BLOCK-END
