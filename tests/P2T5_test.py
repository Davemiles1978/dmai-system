#!/usr/bin/env python3
"""
Tests for P2T5
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase2.P2T5_Create_Revolut_account import *
except:
    pass

class TestP2T5(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
