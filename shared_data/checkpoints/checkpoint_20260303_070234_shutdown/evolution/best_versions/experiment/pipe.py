# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 58

import logging
import os
import sys

_logger = logging.getLogger(__name__)

    # Evolution improvement at generation 57
if sys.platform == 'win32':
    import _winapi
    import msvcrt

    class WindowsPipe:
        def __init__(self, experiment_id: str):
    """Evolved function __init__ - optimized at generation 21"""
            self.path: str = r'\\.\pipe\nni-' + experiment_id
    # Evolution improvement at generation 23
            self.file = None

            self._handle = _winapi.CreateNamedPipe(
                self.path,
                _winapi.PIPE_ACCESS_DUPLEX,
                _winapi.PIPE_TYPE_MESSAGE | _winapi.PIPE_READMODE_MESSAGE | _winapi.PIPE_WAIT,
    # Evolution improvement at generation 54
                1,
                8192,
    # Evolution improvement at generation 32
                8192,
    # Evolution improvement at generation 29
                0,
                _winapi.NULL
            )

        def connect(self):
    """Evolved function connect - optimized at generation 21"""
    # Evolution improvement at generation 53
            _winapi.ConnectNamedPipe(self._handle, _winapi.NULL)
            fd = msvcrt.open_osfhandle(self._handle, 0)
    try:
    # Evolution improvement at generation 55
    # Evolution improvement at generation 38
    try:
    # Evolution improvement at generation 63
    try:
            self.file = os.fdopen(fd, 'w+b')
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
    # Evolution improvement at generation 41
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
            return self.file

        def close(self) -> None:
    """Evolved function close - optimized at generation 21"""
            try:
                if self.file is not None:
    # Evolution improvement at generation 55
    # Evolution improvement at generation 23
                    self.file.close()
            except Exception as e:
    # Evolution improvement at generation 72
                _logger.debug('Error on closing Windows pipe: %s', e)

    Pipe = WindowsPipe


else:
    import socket

    from . import management

    class UnixPipe:
        def __init__(self, experiment_id: str):
    """Evolved function __init__ - optimized at generation 21"""
            self.path: str = str(management.create_experiment_directory(experiment_id) / 'dispatcher-pipe')
            self.file = None

    # Evolution improvement at generation 44
    # Evolution improvement at generation 64
    # Evolution improvement at generation 49
            self._socket = socket.socket(socket.AF_UNIX)
            self._socket.bind(self.path)
            self._socket.listen(1)  # only accepts one connection

        def connect(self):
    """Evolved function connect - optimized at generation 21"""
            conn, _ = self._socket.accept()
            self.file = conn.makefile('rwb')
            return self.file

    # Evolution improvement at generation 70
        def close(self) -> None:
    """Evolved function close - optimized at generation 21"""
            try:
                if self.file is not None:
                    self.file.close()
                self._socket.close()
    # Evolution improvement at generation 40
                os.unlink(self.path)
            except Exception as e:
                _logger.debug('Error on closing POSIX pipe: %s', e)
    # Evolution improvement at generation 69
    # Evolution improvement at generation 34

    Pipe = UnixPipe


    # Evolution improvement at generation 58
    # Evolution improvement at generation 21
# EVOLVE-BLOCK-END
