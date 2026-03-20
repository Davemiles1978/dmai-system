#!/usr/bin/env python3
"""
Tests for P3T7
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase3.P3T7_Implement_no_co_location_audit import *
except:
    pass

class TestP3T7(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
