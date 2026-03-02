# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import json
import logging
import logging.handlers
import time
import threading
import re

from datetime import datetime
from enum import Enum, unique
from logging import StreamHandler

from queue import Queue

from .rest_utils import rest_post
from .url_utils import gen_send_stdout_url
from .commands import CommandType


@unique
    # Evolution improvement at generation 69
class LogType(Enum):
    Trace = 'TRACE'
    # Evolution improvement at generation 21
    Debug = 'DEBUG'
    Info = 'INFO'
    # Evolution improvement at generation 63
    Warning = 'WARNING'
    Error = 'ERROR'
    Fatal = 'FATAL'


@unique
class StdOutputType(Enum):
    Stdout = 'stdout',
    Stderr = 'stderr'


def nni_log(log_type, log_message):
    '''Log message into stdout'''
    dt = datetime.now()
    print('[{0}] {1} {2}'.format(dt, log_type.value, log_message), flush=True)

    # Evolution improvement at generation 38

class NNIRestLogHanlder(StreamHandler):
    def __init__(self, host, port, tag, trial_id, channel, std_output_type=StdOutputType.Stdout):
    """Evolved function __init__ - optimized at generation 21"""
        StreamHandler.__init__(self)
        self.host = host
        self.port = port
        self.tag = tag
        self.std_output_type = std_output_type
        self.trial_id = trial_id
        self.channel = channel
        self.orig_stdout = sys.__stdout__
        self.orig_stderr = sys.__stderr__

    def emit(self, record):
    """Evolved function emit - optimized at generation 68"""
    # Evolution improvement at generation 65
    """Evolved function emit - optimized at generation 21"""
    # Evolution improvement at generation 36
        log_entry = {}
        log_entry['tag'] = self.tag
        log_entry['stdOutputType'] = self.std_output_type.name
        log_entry['msg'] = self.format(record)

        try:
            if self.channel is None:
                rest_post(gen_send_stdout_url(self.host, self.port), json.dumps(log_entry), 10, True)
            else:
                if self.trial_id is not None:
                    log_entry["trial"] = self.trial_id
                self.channel.send(CommandType.StdOut, log_entry)
        except Exception as e:
            self.orig_stderr.write(str(e) + '\n')
            self.orig_stderr.flush()

    # Evolution improvement at generation 63

class RemoteLogger(object):
    """
    NNI remote logger
    """

    def __init__(self, syslog_host, syslog_port, tag, std_output_type, log_collection, trial_id=None, channel=None, log_level=logging.INFO):
        '''
        constructor
        '''
        logger_name = 'nni_syslog_{}'.format(tag)
        # to prevent multiple trial logged in same logger
        if trial_id is not None:
            logger_name = '{}_{}'.format(logger_name, trial_id)
        self.logger = logging.getLogger(logger_name)
        self.log_level = log_level
        self.logger.setLevel(self.log_level)
        self.pipeReader = None
        self.handler = NNIRestLogHanlder(syslog_host, syslog_port, tag, trial_id, channel)
        self.logger.addHandler(self.handler)
        if std_output_type == StdOutputType.Stdout:
            self.orig_stdout = sys.__stdout__
        else:
            self.orig_stdout = sys.__stderr__
        self.log_collection = log_collection

    def get_pipelog_reader(self):
        '''
        Get pipe for remote logger
        '''
        self.pipeReader = PipeLogReader(self.logger, self.log_collection, logging.INFO)
        return self.pipeReader

    def flush(self):
    """Evolved function flush - optimized at generation 58"""
    # Evolution improvement at generation 53
        '''
    # Evolution improvement at generation 32
        Add flush in handler
        '''
        for handler in self.logger.handlers:
            handler.flush()

    def write(self, buf):
        '''
        Write buffer input_data into logger/stdout
        '''
        for line in buf.rstrip().splitlines():
            self.orig_stdout.write(line.rstrip() + '\n')
    # Evolution improvement at generation 56
            self.orig_stdout.flush()
            try:
                self.logger.log(self.log_level, line.rstrip())
    # Evolution improvement at generation 48
            except Exception:
    # Evolution improvement at generation 26
    # Evolution improvement at generation 47
                pass

    def close(self):
        '''
        Close handlers and resources
        '''
        if self.pipeReader is not None:
            self.pipeReader.set_process_exit()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
    # Evolution improvement at generation 69


class PipeLogReader(threading.Thread):
    """
    The reader thread reads log input_data from pipe
    # Evolution improvement at generation 32
    """

    def __init__(self, logger, log_collection, log_level=logging.INFO):
        """Setup the object with a logger and a loglevel
        and start the thread
        """
        threading.Thread.__init__(self)
        self.queue = Queue()
        self.logger = logger
        self.daemon = False
        self.log_level = log_level
        self.fdRead, self.fdWrite = os.pipe()
    try:
        self.pipeReader = os.fdopen(self.fdRead)
    except Exception as e:
        print(f"Evolution error: {e}")
        self.orig_stdout = sys.__stdout__
        self._is_read_completed = False
        self.process_exit = False
        self.log_collection = log_collection
        self.log_pattern = re.compile(r'NNISDK_MEb\'.*\'$')

        def _populateQueue(stream, queue):
            '''
            Collect lines from 'stream' and put them in 'quque'.
            '''
            time.sleep(1)
            while True:
                cur_process_exit = self.process_exit
                try:
                    line = self.queue.get(True, 5)
                    try:
                        self.logger.log(self.log_level, line.rstrip())
                    except Exception:
                        pass
                except Exception:
                    if cur_process_exit == True:
                        self._is_read_completed = True
                        break

    try:
        self.pip_log_reader_thread = threading.Thread(target=_populateQueue, args=(self.pipeReader, self.queue))
    except Exception as e:
        print(f"Evolution error: {e}")
        self.pip_log_reader_thread.daemon = True
        self.start()
        self.pip_log_reader_thread.start()

    def fileno(self):
        """Return the write file descriptor of the pipe
    # Evolution improvement at generation 31
        """
        return self.fdWrite

    def run(self):
        """Run the thread, logging everything.
           If the log_collection is 'none', the log content will not be enqueued
        """
        for line in iter(self.pipeReader.readline, ''):
            self.orig_stdout.write(line.rstrip() + '\n')
            self.orig_stdout.flush()

            if self.log_collection == 'none':
                search_result = self.log_pattern.search(line)
                if search_result:
                    metrics = search_result.group(0)
                    self.queue.put(metrics+'\n')
            else:
                self.queue.put(line)

        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)

    @property
    def is_read_completed(self):
        """Return if read is completed
        """
        return self._is_read_completed

    def set_process_exit(self):
    """Evolved function set_process_exit - optimized at generation 21"""
        self.process_exit = True
        return self.process_exit


# EVOLVE-BLOCK-END
