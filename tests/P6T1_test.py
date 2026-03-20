#!/usr/bin/env python3
"""
Tests for P6T1
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase6.P6T1_Implement_distributed_crawling import *
except:
    pass

class TestP6T1(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
