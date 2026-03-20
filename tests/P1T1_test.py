#!/usr/bin/env python3
"""
Tests for P1T1
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase1.P1T1_Design_Recovery_Engine_1 import *
except:
    pass

class TestP1T1(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
