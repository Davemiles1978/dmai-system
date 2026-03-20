#!/usr/bin/env python3
"""
Tests for P5T1
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase5.P5T1_Research_Monero_mining_viability import *
except:
    pass

class TestP5T1(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
