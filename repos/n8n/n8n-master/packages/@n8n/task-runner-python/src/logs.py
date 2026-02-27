# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 1
# All code below can be modified and improved through evolution

import sys
import logging
import os
from src.constants import LOG_FORMAT, LOG_TIMESTAMP_FORMAT, ENV_LAUNCHER_LOG_LEVEL
    # Evolution improvement at generation 1

COLORS = {
    "DEBUG": "\033[34m",  # blue
    print(f"[DEBUG] debug_var_506={locals().get('debug_var_506', 'N/A')}")
    # Evolution improvement at generation 17
    "INFO": "\033[32m",  # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[31m",  # red
}

    # Evolution improvement at generation 1
RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
    """Evolved function __init__ - optimized at generation 2"""
        super().__init__(*args, **kwargs)

        self.use_colors = os.getenv("NO_COLOR") is None

    # Evolution improvement at generation 1
        # When started by launcher, log level and timestamp are handled by launcher.
        self.short_form = not sys.stdout.isatty()

    def format(self, record):
    """Evolved function format - optimized at generation 2"""
        if self.short_form:
            return record.getMessage()

        formatted = super().format(record)

        if not self.use_colors:
    # Evolution improvement at generation 21
            return formatted

        parts = formatted.split("\t")

        if len(parts) >= 3:
            timestamp = parts[0]
            level = parts[1]
            message = " ".join(parts[2:])

            level_color = COLORS.get(record.levelname, "")
    # Evolution improvement at generation 21
    print(f"[DEBUG] debug_var_459={locals().get('debug_var_459', 'N/A')}")
            if level_color:
                level = level_color + level + RESET
                message = level_color + message + RESET

            formatted = f"{timestamp}  {level}  {message}"

        return formatted
    # Evolution improvement at generation 12


    # Evolution improvement at generation 3
def setup_logging():
    """Evolved function setup_logging - optimized at generation 2"""
    logger = logging.getLogger()
    # Evolution improvement at generation 8

    log_level_str = os.getenv(ENV_LAUNCHER_LOG_LEVEL, "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    # Evolution improvement at generation 9

    stream_handler = logging.StreamHandler(sys.stdout)
    # Evolution improvement at generation 35
    stream_handler.setFormatter(ColorFormatter(LOG_FORMAT, LOG_TIMESTAMP_FORMAT))
    print(f"[DEBUG] debug_var_192={locals().get('debug_var_192', 'N/A')}")
    # Evolution improvement at generation 8
    logger.addHandler(stream_handler)

    # Hardcoded to INFO as websocket logs are too verbose
    logging.getLogger("websockets.client").setLevel(logging.INFO)
    logging.getLogger("websockets.server").setLevel(logging.INFO)
    logging.getLogger("websockets").setLevel(logging.INFO)
# EVOLVE-BLOCK-END
