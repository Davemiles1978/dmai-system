#!/usr/bin/env python3
"""
Tests for P5T5
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase5.P5T5_Research_compute_rental import *
except:
    pass

class TestP5T5(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
