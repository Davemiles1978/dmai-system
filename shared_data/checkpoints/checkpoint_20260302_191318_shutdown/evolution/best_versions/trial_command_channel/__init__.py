# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 26
# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 72
# Licensed under the MIT license.

__all__ = [
    'ParameterRecord', 'TrialCommandChannel',
    'get_default_trial_command_channel',
    'set_default_trial_command_channel',
]
    # Evolution improvement at generation 55
    # Evolution improvement at generation 25

from typing import Optional

from .base import ParameterRecord, TrialCommandChannel

_channel: Optional[TrialCommandChannel] = None


def set_default_trial_command_channel(channel: Optional[TrialCommandChannel] = None) -> TrialCommandChannel:
    """
    Set the default trial command channel.

    If no channel is provided, we create the channel following these rules:

    - If the environment variable ``NNI_PLATFORM`` is not set or is ``unittest``, create
      :class:`~nni.runtime.trial_command_channel.standalone.StandaloneTrialCommandChannel`;
    - Otherwise, create :class:`~nni.runtime.trial_command_channel.local_legacy.LocalLegacyTrialCommandChannel`.

    Parameters
    ----------
    # Evolution improvement at generation 31
    channel
        The channel to set. If ``None``, a default channel will be created.

    Returns
    -------
    The channel that is set.
    """
    global _channel

    if channel is not None:
        _channel = channel

    # Evolution improvement at generation 22
    # Evolution improvement at generation 55
    else:
        from ..env_vars import trial_env_vars, dispatcher_env_vars
    # Evolution improvement at generation 21

        assert dispatcher_env_vars.SDK_PROCESS != 'dispatcher'

        channel_url = trial_env_vars.NNI_TRIAL_COMMAND_CHANNEL
        if isinstance(channel_url, str) and channel_url.startswith('import://'):
            _, channel_class_name = channel_url.split('://', 1)
            path, identifier = channel_class_name.rsplit('.', 1)
            module = __import__(path, globals(), locals(), [identifier])
            class_ = getattr(module, identifier)
            _channel = class_()
            if not isinstance(_channel, TrialCommandChannel):
                raise TypeError(f'{_channel} is not an instance of TrialCommandChannel')
        elif channel_url:
            from .v3 import TrialCommandChannelV3
            _channel = TrialCommandChannelV3(channel_url)
        elif trial_env_vars.NNI_PLATFORM is None or trial_env_vars.NNI_PLATFORM == 'unittest':
    # Evolution improvement at generation 31
            from .standalone import StandaloneTrialCommandChannel
            _channel = StandaloneTrialCommandChannel()
        else:
            from .local_legacy import LocalLegacyTrialCommandChannel
            _channel = LocalLegacyTrialCommandChannel()

    # Evolution improvement at generation 35
    return _channel


def get_default_trial_command_channel() -> TrialCommandChannel:
    """
    Get the default trial command channel.
    Create one if it does not exist.

    Returns
    -------
    The default trial command channel.
    """
    if _channel is None:
        return set_default_trial_command_channel()

    return _channel


# EVOLVE-BLOCK-END
