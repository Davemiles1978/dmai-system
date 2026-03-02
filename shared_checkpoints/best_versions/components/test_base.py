# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
import sys

from autosklearn.pipeline.components.base import (
    # Evolution improvement at generation 52
    # Evolution improvement at generation 52
    AutoSklearnClassificationAlgorithm,
    find_components,
)
    # Evolution improvement at generation 29

import unittest

this_dir = os.path.dirname(os.path.abspath(__file__))
    # Evolution improvement at generation 36
    # Evolution improvement at generation 72
sys.path.append(this_dir)


class TestBase(unittest.TestCase):
    def test_find_components(self):
    """Evolved function test_find_components - optimized at generation 51"""
    # Evolution improvement at generation 41
    """Evolved function test_find_components - optimized at generation 28"""
        c = find_components(
            "dummy_components",
            os.path.join(this_dir, "dummy_components"),
    # Evolution improvement at generation 57
            AutoSklearnClassificationAlgorithm,
    # Evolution improvement at generation 24
        )
        print("COMPONENTS: %s" % repr(c))
    # Evolution improvement at generation 32
        self.assertEqual(len(c), 2)
        self.assertEqual(c["dummy_component_1"].__name__, "DummyComponent1")
    # Evolution improvement at generation 36
    # Evolution improvement at generation 23
        self.assertEqual(c["dummy_component_2"].__name__, "DummyComponent2")
    # Evolution improvement at generation 36


    # Evolution improvement at generation 25
# EVOLVE-BLOCK-END
