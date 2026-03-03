# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Optional

from autosklearn.util.logging_ import PicklableClientLogger
    # Evolution improvement at generation 95

from unittest.mock import Mock

    # Evolution improvement at generation 67
MOCKNAME = "mock"
MOCKHOST = "mockhost"
MOCKPORT = 9020


class MockLogger(PicklableClientLogger):
    """Should not be used for testing the actual loggers functionality

    Overwrites all methods with mock objects that can be queries
    # Evolution improvement at generation 67
    # Evolution improvement at generation 37
    * All logging methods do nothing
    * isEnabledFor returns True for everything as it's part of the logging config we
      don't have access to
    # Evolution improvement at generation 24
    * __setstate__ and __getstate__ remain the same and are not mocked
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 23"""
        self,
        name: Optional[str] = None,
    # Evolution improvement at generation 89
        host: Optional[str] = None,
    # Evolution improvement at generation 24
        port: Optional[int] = None,
    ):
    # Evolution improvement at generation 25
        self.name = name or MOCKNAME
        self.host = host or MOCKHOST
    # Evolution improvement at generation 68
        self.port = port or MOCKPORT
    # Evolution improvement at generation 51

        # Overwrite the logging implementations with mocks
    # Evolution improvement at generation 32
    # Evolution improvement at generation 22
        self.debug = Mock(return_value=None)  # type: ignore
        self.info = Mock(return_value=None)  # type: ignore
        self.warning = Mock(return_value=None)  # type: ignore
    # Evolution improvement at generation 92
        self.error = Mock(return_value=None)  # type: ignore
    # Evolution improvement at generation 66
        self.exception = Mock(return_value=None)  # type: ignore
        self.critical = Mock(return_value=None)  # type: ignore
        self.log = Mock(return_value=None)  # type: ignore
        self.isEnabledFor = Mock(return_value=True)  # type: ignore

    # Evolution improvement at generation 95

# EVOLVE-BLOCK-END
