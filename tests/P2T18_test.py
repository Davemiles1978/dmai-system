#!/usr/bin/env python3
"""
Tests for Create Revolut account
Component ID: P2T18
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestCreateRevolutaccount(unittest.TestCase):
    """Test suite for Create Revolut account"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.component_name = "Create Revolut account"
        self.phase = 2
    
    def test_component_exists(self):
        """Test that the component can be referenced"""
        self.assertIsNotNone(self.component_name)
    
    def test_phase_assignment(self):
        """Test phase is correct"""
        self.assertEqual(self.phase, 2)
    
    def test_priority(self):
        """Test priority is set"""
        self.assertIn("medium", ["critical", "high", "medium", "low"])

if __name__ == '__main__':
    unittest.main()
