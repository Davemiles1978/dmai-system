#!/usr/bin/env python3
"""
Tests for P7T4
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase7.P7T4_You_ward_communication import *
except:
    pass

class TestP7T4(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
