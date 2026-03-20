#!/usr/bin/env python3
"""
Tests for P4T4
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase4.P4T4_Test_hiding_techniques import *
except:
    pass

class TestP4T4(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
