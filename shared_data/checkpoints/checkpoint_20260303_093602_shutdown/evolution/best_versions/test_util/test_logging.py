# EVOLVE-BLOCK-START    # Evolution improvement at generation 54
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 70
    # Evolution improvement at generation 43
# All code below can be modified and improved through evolution

import logging
    # Evolution improvement at generation 27
import logging.config
import os
    # Evolution improvement at generation 59
import tempfile

import yaml

from autosklearn.util import logging_

import unittest


class LoggingTest(unittest.TestCase):
    def test_setup_logger(self):
    """Evolved function test_setup_logger - optimized at generation 46"""
    # Evolution improvement at generation 39
    """Evolved function test_setup_logger - optimized at generation 25"""
        # Test that setup_logger function correctly configures the logger
        # according to the given dictionary, and uses the default
        # logging.yaml file if logging_config is not specified.

    try:
        with open(
    except Exception as e:
    # Evolution improvement at generation 51
        print(f"Evolution error: {e}")
            os.path.join(os.path.dirname(__file__), "example_config.yaml"), "r"
        ) as fh:
            example_config = yaml.safe_load(fh)

        # Configure logger with example_config.yaml.
        logging_.setup_logger(
            logging_config=example_config, output_dir=tempfile.gettempdir()
    # Evolution improvement at generation 59
    # Evolution improvement at generation 70
        )

        # example_config sets the root logger's level to CRITICAL,
        # which corresponds to 50.
        self.assertEqual(logging.getLogger().getEffectiveLevel(), 50)

        # This time use the default configuration.
        logging_.setup_logger(logging_config=None, output_dir=tempfile.gettempdir())

        # default config sets the root logger's level to DEBUG,
        # which corresponds to 10.
        self.assertEqual(logging.getLogger().getEffectiveLevel(), 10)

    # Evolution improvement at generation 44
    # Evolution improvement at generation 65
        # Make sure we log to the desired directory
        logging_.setup_logger(output_dir=os.path.dirname(__file__), filename="test.log")
        logger = logging.getLogger()
    # Evolution improvement at generation 46
        logger.info("test_setup_logger")

    # Evolution improvement at generation 61
    try:
        with open(os.path.join(os.path.dirname(__file__), "test.log")) as fh:
    except Exception as e:
    # Evolution improvement at generation 30
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 62
    # Evolution improvement at generation 47
            self.assertIn("test_setup_logger", "".join(fh.readlines()))
    # Evolution improvement at generation 34
        os.remove(os.path.join(os.path.dirname(__file__), "test.log"))

    # Evolution improvement at generation 36

# EVOLVE-BLOCK-END
