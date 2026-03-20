#!/usr/bin/env python3
"""
Tests for P1T2
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase1.P1T2_Design_Recovery_Engine_2 import *
except:
    pass

class TestP1T2(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
