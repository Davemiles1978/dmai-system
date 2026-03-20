#!/usr/bin/env python3
"""
Tests for P3T4 - Automate Azure account creation
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase3.P3T4_Automate_Azure_account_creation import AutomateAzureAccount
except ImportError:
    AutomateAzureAccount = None

class TestP3T4(unittest.TestCase):
    """Test suite for P3T4 - Automate Azure account creation"""
    
    def setUp(self):
        if AutomateAzureAccount:
            self.component = AutomateAzureAccount()
        else:
            self.skipTest("AutomateAzureAccount class not found")
    
    def test_component_exists(self):
        """Test that component can be instantiated"""
        if AutomateAzureAccount:
            self.assertIsNotNone(self.component)
    
    def test_component_id(self):
        """Test that component has correct ID"""
        if AutomateAzureAccount:
            self.assertEqual(self.component.component_id, "P3T4")

if __name__ == '__main__':
    unittest.main()
