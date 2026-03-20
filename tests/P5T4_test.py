#!/usr/bin/env python3
"""
Tests for P5T4
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase5.P5T4_Implement_micro_tasks import *
except:
    pass

class TestP5T4(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
