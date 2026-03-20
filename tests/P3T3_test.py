#!/usr/bin/env python3
"""
Tests for P3T3 - Automate GCP account creation
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase3.P3T3_Automate_GCP_account_creation import AutomateGCPAccount
except ImportError:
    AutomateGCPAccount = None

class TestP3T3(unittest.TestCase):
    """Test suite for P3T3 - Automate GCP account creation"""
    
    def setUp(self):
        if AutomateGCPAccount:
            self.component = AutomateGCPAccount()
        else:
            self.skipTest("AutomateGCPAccount class not found")
    
    def test_component_exists(self):
        """Test that component can be instantiated"""
        if AutomateGCPAccount:
            self.assertIsNotNone(self.component)
    
    def test_component_id(self):
        """Test that component has correct ID"""
        if AutomateGCPAccount:
            self.assertEqual(self.component.component_id, "P3T3")

if __name__ == '__main__':
    unittest.main()
