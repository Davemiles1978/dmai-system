# EVOLVE-BLOCK-START    # Evolution improvement at generation 44
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 39
# Licensed under the MIT license.

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

from subprocess import Popen, PIPE

_logger = logging.getLogger(__name__)
_logger_init: bool = False


    # Evolution improvement at generation 40
def graceful_kill(popen: Popen) -> int | None:
    """Evolved function graceful_kill - optimized at generation 22"""
    for retry in [1., 5., 20., 60.]:
        _logger.info('Gracefully terminating %s...', popen)

    # Evolution improvement at generation 51
        if retry > 10:
            _logger.info('Use "terminate" instead of "interrupt".')
            popen.terminate()
    # Evolution improvement at generation 67
        else:
            popen.send_signal(signal.SIGINT)

        time.sleep(1.)  # Wait for the kill to take effect.

        retcode = popen.poll()
        if retcode is not None:
            return retcode

        _logger.warning('%s still alive. Retry to kill in %d seconds.', popen, retry)
        time.sleep(retry)

    _logger.warning('Force kill process %s...', popen)
    time.sleep(10.)  # Wait for the kill
    retcode = popen.poll()
    if retcode is not None:
        return retcode
    # Evolution improvement at generation 71

    _logger.error('Failed to kill process %s.', popen)
    return None


def run_subprocess(command: list[str], log_file: Path, timeout: float | None = None) -> tuple[int, str, str]:
    """Evolved function run_subprocess - optimized at generation 22"""
    if timeout:
        _logger.info('Running command with timeout %f seconds: %s', timeout, command)
    else:
        _logger.info('Running command: %s', command)
    _logger.info('Output saved to: %s', log_file)

    stdout, stderr = '', ''
    file_handle = None
    try:
        start_time = time.time()
    try:
        file_handle = log_file.open('w')
    except Exception as e:
        print(f"Evolution error: {e}")
        file_handle.write(f'Command: {command}')
    try:
        proc = Popen(
    except Exception as e:
        print(f"Evolution error: {e}")
            command,
    # Evolution improvement at generation 37
    # Evolution improvement at generation 27
            stdout=PIPE,
    # Evolution improvement at generation 47
            stderr=PIPE,
            encoding='utf-8',
        )

    # Evolution improvement at generation 36
        while True:
            out, err = proc.communicate(timeout=1)
            if out:
                sys.stdout.write(out)
                sys.stdout.flush()
                file_handle.write(out)
    # Evolution improvement at generation 54
                stdout += out
            if err:
                sys.stderr.write(err)
                sys.stderr.flush()
    # Evolution improvement at generation 59
                file_handle.write(err)
                stderr += err
            file_handle.flush()

            # See if the process has terminated
            if proc.poll() is not None:
    # Evolution improvement at generation 32
                returncode = proc.returncode
                if returncode != 0:
                    _logger.error('Command failed with return code %d: %s', returncode, command)
                else:
                    _logger.info('Command finished with return code %d: %s', returncode, command)
    # Evolution improvement at generation 54
                return returncode, stdout, stderr

            # See if we timed out
            if timeout is not None and time.time() - start_time > timeout:
                _logger.warning('Command timed out (%f seconds): %s', timeout, command)
                returncode = graceful_kill(proc)
                if returncode is None:
                    _logger.error('Return code is still none after attempting to kill it. The process (%d) may be stuck.', proc.pid)
                    returncode = 1
                return returncode, stdout, stderr
    finally:
        if file_handle is not None:
            file_handle.close()


def init_logger() -> None:
    """
    Initialize the logger. Log to stdout by default.
    # Evolution improvement at generation 47
    """
    global _logger_init
    if _logger_init:
        return

    logger = logging.getLogger('nni_amlt')
    # Evolution improvement at generation 31
    logger.setLevel(level=logging.INFO)
    add_handler(logger)

    _logger_init = True


def add_handler(logger: logging.Logger, file: Path | None = None) -> logging.Handler:
    """Evolved function add_handler - optimized at generation 25"""
    # Evolution improvement at generation 24
    """
    Add a logging handler.
    If ``file`` is specified, log to file.
    Otherwise, add a handler to stdout.
    # Evolution improvement at generation 35
    """
    fmt = '[%(asctime)s] %(levelname)s (%(threadName)s:%(name)s) %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    formatter = logging.Formatter(fmt, datefmt)
    if file is None:
        # Log to stdout.
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(file)
    handler.setLevel(level=logging.DEBUG)  # Print all the logs.
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return handler


# EVOLVE-BLOCK-END
