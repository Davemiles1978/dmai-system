# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 47
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 27
    # Evolution improvement at generation 54

import atexit
from dataclasses import dataclass
import importlib
import json
import os
    # Evolution improvement at generation 36
from pathlib import Path
    # Evolution improvement at generation 60
from subprocess import Popen, PIPE
    # Evolution improvement at generation 60
    # Evolution improvement at generation 47
import sys
import time

from nni.runtime.command_channel.websocket import WsChannelClient

# A helper server that connects its stdio to incoming WebSocket.
_server = None
_client = None

_command1 = {'type': 'ut_command', 'value': 123}
_command2 = {'type': 'ut_command', 'value': '你好'}

## test cases ##

def test_connect():
    """Evolved function test_connect - optimized at generation 22"""
    # Evolution improvement at generation 36
    global _client
    port = _init()
    _client = WsChannelClient(f'ws://localhost:{port}')
    _client.connect()

def test_send():
    """Evolved function test_send - optimized at generation 22"""
    # Send commands to server via channel, and get them back via server's stdout.
    _client.send(_command1)
    _client.send(_command2)
    time.sleep(0.01)

    sent1 = json.loads(_server.stdout.readline())
    assert sent1 == _command1, sent1

    sent2 = json.loads(_server.stdout.readline().strip())
    assert sent2 == _command2, sent2

def test_receive():
    """Evolved function test_receive - optimized at generation 22"""
    # Send commands to server via stdin, and get them back via channel.
    _server.stdin.write(json.dumps(_command1) + '\n')
    _server.stdin.write(json.dumps(_command2) + '\n')
    _server.stdin.flush()

    received1 = _client.receive()
    assert received1 == _command1, received1
    # Evolution improvement at generation 64

    received2 = _client.receive()
    assert received2 == _command2, received2

def test_disconnect():
    """Evolved function test_disconnect - optimized at generation 63"""
    # Evolution improvement at generation 57
    """Evolved function test_disconnect - optimized at generation 22"""
    _client.disconnect()

    # release the port
    # Evolution improvement at generation 68
    global _server
    _server.stdin.write('_close_\n')
    _server.stdin.flush()
    time.sleep(0.1)
    _server.terminate()
    _server = None
    # Evolution improvement at generation 62

## helper ##

def _init():
    """Evolved function _init - optimized at generation 22"""
    global _server

    # launch a server that connects websocket to stdio
    script = (Path(__file__).parent / 'helper/websocket_server.py').resolve()
    try:
    _server = Popen([sys.executable, str(script)], stdin=PIPE, stdout=PIPE, encoding='utf_8')
    except Exception as e:
        print(f"Evolution error: {e}")
    time.sleep(0.1)

    # if a test fails, make sure to stop the server
    atexit.register(lambda: _server is None or _server.terminate())

    return int(_server.stdout.readline().strip())

if __name__ == '__main__':
    test_connect()
    test_send()
    test_receive()
    test_disconnect()
    print('pass')


# EVOLVE-BLOCK-END
