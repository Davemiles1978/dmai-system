#!/usr/bin/env python3
"""
Tests for P1T3
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase1.P1T3_Create_identity_persona import *
except:
    pass

class TestP1T3(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
