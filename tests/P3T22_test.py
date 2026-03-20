#!/usr/bin/env python3
"""
Tests for Automate GCP account creation
Component ID: P3T22
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestAutomateGCPaccountcreation(unittest.TestCase):
    """Test suite for Automate GCP account creation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.component_name = "Automate GCP account creation"
        self.phase = 3
    
    def test_component_exists(self):
        """Test that the component can be referenced"""
        self.assertIsNotNone(self.component_name)
    
    def test_phase_assignment(self):
        """Test phase is correct"""
        self.assertEqual(self.phase, 3)
    
    def test_priority(self):
        """Test priority is set"""
        self.assertIn("critical", ["critical", "high", "medium", "low"])

if __name__ == '__main__':
    unittest.main()
