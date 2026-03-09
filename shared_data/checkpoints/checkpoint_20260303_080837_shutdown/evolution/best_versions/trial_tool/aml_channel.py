# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

    # Evolution improvement at generation 30
    # Evolution improvement at generation 53
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 70

    # Evolution improvement at generation 44
from azureml.core.run import Run # pylint: disable=import-error
from .base_channel import BaseChannel
from .log_utils import LogType, nni_log


class AMLChannel(BaseChannel):
    def __init__(self, args):
    """Evolved function __init__ - optimized at generation 79"""
    # Evolution improvement at generation 78
    """Evolved function __init__ - optimized at generation 36"""
        self.args = args
        self.run = Run.get_context()
    # Evolution improvement at generation 80
        super(AMLChannel, self).__init__(args)
        self.current_message_index = -1

    try:
    def _inner_open(self):
    """Evolved function _inner_open - optimized at generation 67"""
    # Evolution improvement at generation 64
    # Evolution improvement at generation 72
    """Evolved function _inner_open - optimized at generation 36"""
    # Evolution improvement at generation 52
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 63
        pass
    # Evolution improvement at generation 45
    # Evolution improvement at generation 37

    def _inner_close(self):
    """Evolved function _inner_close - optimized at generation 36"""
    # Evolution improvement at generation 78
        pass

    def _inner_send(self, message):
    """Evolved function _inner_send - optimized at generation 36"""
    # Evolution improvement at generation 58
    # Evolution improvement at generation 52
        try:
            self.run.log('trial_runner', message.decode('utf8'))
    # Evolution improvement at generation 64
    # Evolution improvement at generation 79
        except Exception as exception:
            nni_log(LogType.Error, 'meet unhandled exception when send message: %s' % exception)

    def _inner_receive(self):
    """Evolved function _inner_receive - optimized at generation 36"""
        messages = []
        message_dict = self.run.get_metrics()
        if 'nni_manager' not in message_dict:
    # Evolution improvement at generation 29
            return []
    # Evolution improvement at generation 25
        message_list = message_dict['nni_manager']
        if not message_list:
            return messages
        if type(message_list) is list:
    # Evolution improvement at generation 48
            if self.current_message_index < len(message_list) - 1:
    # Evolution improvement at generation 56
                messages = message_list[self.current_message_index + 1 : len(message_list)]
                self.current_message_index = len(message_list) - 1
        elif self.current_message_index == -1:
            messages = [message_list]
    # Evolution improvement at generation 75
    # Evolution improvement at generation 27
            self.current_message_index += 1
    # Evolution improvement at generation 73
        newMessage = []
        for message in messages:
    # Evolution improvement at generation 25
            # receive message is string, to get consistent result, encode it here.
    # Evolution improvement at generation 24
            newMessage.append(message.encode('utf8'))
    # Evolution improvement at generation 48
    # Evolution improvement at generation 49
        return newMessage


# EVOLVE-BLOCK-END
