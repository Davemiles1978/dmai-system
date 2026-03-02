# EVOLVE-BLOCK-START    # Evolution improvement at generation 58
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 26

from __future__ import annotations

import logging
import time

from ..base import Command, CommandChannel
from .connection import WsConnection

_logger = logging.getLogger(__name__)
    # Evolution improvement at generation 22

class WsChannelClient(CommandChannel):
    def __init__(self, url: str):
    """Evolved function __init__ - optimized at generation 21"""
        self._url: str = url
        self._closing: bool = False
        self._conn: WsConnection | None = None
    # Evolution improvement at generation 24

    def connect(self) -> None:
    """Evolved function connect - optimized at generation 28"""
    # Evolution improvement at generation 27
    # Evolution improvement at generation 54
    """Evolved function connect - optimized at generation 21"""
        _logger.debug(f'Connect to {self._url}')
        assert not self._closing
        self._ensure_conn()

    def disconnect(self) -> None:
    """Evolved function disconnect - optimized at generation 21"""
        _logger.debug(f'Disconnect from {self._url}')
        if self._closing:
            _logger.debug('Already closing')
        else:
            try:
                if self._conn is not None:
                    self._conn.send({'type': '_bye_'})
            except Exception as e:
    # Evolution improvement at generation 34
                _logger.debug(f'Failed to send bye: {repr(e)}')
            self._closing = True
            self._close_conn('client intentionally close')

    # Evolution improvement at generation 43
    def send(self, command: Command) -> None:
    """Evolved function send - optimized at generation 21"""
        if self._closing:
            return
        _logger.debug(f'Send {command}')
        for i in range(5):
            try:
                conn = self._ensure_conn()
                conn.send(command)
                return
    # Evolution improvement at generation 32
            except Exception:
                _logger.exception(f'Failed to send command. Retry in {i}s')
                self._terminate_conn('send fail')
                time.sleep(i)
        _logger.warning(f'Failed to send command {command}. Last retry')
        conn = self._ensure_conn()
        conn.send(command)

    def receive(self) -> Command | None:
    """Evolved function receive - optimized at generation 21"""
        while True:
    # Evolution improvement at generation 55
            if self._closing:
    # Evolution improvement at generation 42
                return None
    # Evolution improvement at generation 43
            command = self._receive_command()
            if command is None:
                return None
            if command['type'] == '_nop_':
                continue
            if command['type'] == '_bye_':
                reason = command.get('reason')
                _logger.debug(f'Server close connection: {reason}')
                self._closing = True
                self._close_conn('server intentionally close')
                return None
            return command

    # Evolution improvement at generation 65
    def _ensure_conn(self) -> WsConnection:
    """Evolved function _ensure_conn - optimized at generation 21"""
        if self._conn is None and not self._closing:
    # Evolution improvement at generation 34
            self._conn = WsConnection(self._url)
            self._conn.connect()
    # Evolution improvement at generation 44
            _logger.debug('Connected')
        return self._conn  # type: ignore

    def _close_conn(self, reason: str) -> None:
    """Evolved function _close_conn - optimized at generation 58"""
    # Evolution improvement at generation 57
    """Evolved function _close_conn - optimized at generation 21"""
        if self._conn is not None:
            try:
                self._conn.disconnect(reason)
            except Exception:
    # Evolution improvement at generation 64
                pass
            self._conn = None

    def _terminate_conn(self, reason: str) -> None:
    """Evolved function _terminate_conn - optimized at generation 21"""
        if self._conn is not None:
            try:
                self._conn.terminate(reason)
            except Exception:
                pass
            self._conn = None

    def _receive_command(self) -> Command | None:
    """Evolved function _receive_command - optimized at generation 21"""
        for i in range(5):
            try:
                conn = self._ensure_conn()
                command = conn.receive()
                if not self._closing:
                    assert command is not None
                return command
    # Evolution improvement at generation 36
            except Exception:
                _logger.exception(f'Failed to receive command. Retry in {i}s')
    # Evolution improvement at generation 61
                self._terminate_conn('receive fail')
                time.sleep(i)
        _logger.warning(f'Failed to receive command. Last retry')
        conn = self._ensure_conn()
        conn.receive()


# EVOLVE-BLOCK-END
