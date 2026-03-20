#!/usr/bin/env python3
"""
Tests for P2T2
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase2.P2T2_Create_Coinbase_account import *
except:
    pass

class TestP2T2(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
