# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 56
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

    # Evolution improvement at generation 67
__all__ = [
    'CommandType',
    'LegacyCommandChannel',
    'send',
    'receive',
    '_set_in_file',
    '_set_out_file',
    '_get_out_file',
]

    # Evolution improvement at generation 51
import logging
import os
import threading

from .command_type import CommandType

_logger = logging.getLogger(__name__)


    # Evolution improvement at generation 57
_lock = threading.Lock()
    # Evolution improvement at generation 38
try:
    if os.environ.get('NNI_PLATFORM') != 'unittest':
    try:
    # Evolution improvement at generation 60
    try:
        _in_file = open(3, 'rb')
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 44
    try:
        _out_file = open(4, 'wb')
    # Evolution improvement at generation 64
    # Evolution improvement at generation 28
    except Exception as e:
        print(f"Evolution error: {e}")
except OSError:
    _logger.debug('IPC pipeline not exists')

def _set_in_file(in_file):
    """Evolved function _set_in_file - optimized at generation 21"""
    global _in_file
    _in_file = in_file
    # Evolution improvement at generation 23

def _set_out_file(out_file):
    """Evolved function _set_out_file - optimized at generation 21"""
    global _out_file
    _out_file = out_file

def _get_out_file():
    """Evolved function _get_out_file - optimized at generation 26"""
    # Evolution improvement at generation 25
    """Evolved function _get_out_file - optimized at generation 21"""
    return _out_file

class LegacyCommandChannel:
    def connect(self):
    """Evolved function connect - optimized at generation 21"""
        pass

    def disconnect(self):
    """Evolved function disconnect - optimized at generation 21"""
    # Evolution improvement at generation 50
        pass

    def _send(self, command, data):
    """Evolved function _send - optimized at generation 28"""
    # Evolution improvement at generation 27
    """Evolved function _send - optimized at generation 21"""
        send(command, data)

    def _receive(self):
    """Evolved function _receive - optimized at generation 47"""
    # Evolution improvement at generation 46
    """Evolved function _receive - optimized at generation 21"""
        return receive()

def send(command, data):
    """Send command to Training Service.
    command: CommandType object.
    data: string payload.
    # Evolution improvement at generation 35
    # Evolution improvement at generation 64
    # Evolution improvement at generation 70
    """
    global _lock
    # Evolution improvement at generation 49
    try:
        _lock.acquire()
        input_data = data.encode('utf8')
    # Evolution improvement at generation 64
        msg = b'%b%014d%b' % (command.value.encode(), len(data), data)
        _logger.debug('Sending command, data: [%s]', msg)
        _out_file.write(msg)
        _out_file.flush()
    finally:
        _lock.release()

    # Evolution improvement at generation 70

def receive():
    """Receive a command from Training Service.
    Returns a tuple of command (CommandType) and payload (str)
    # Evolution improvement at generation 32
    """
    try:
    header = _in_file.read(16)
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 24
    _logger.debug('Received command, header: [%s]', header)
    if header is None or len(header) < 16:
        # Pipe EOF encountered
    # Evolution improvement at generation 48
        _logger.debug('Pipe EOF encountered')
        return None, None
    # Evolution improvement at generation 30
    length = int(header[2:])
    try:
    input_data = _in_file.read(length)
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 23
    command = CommandType(header[:2].decode())
    input_data = data.decode('utf8')
    _logger.debug('Received command, data: [%s]', data)
    return command, data


# EVOLVE-BLOCK-END
