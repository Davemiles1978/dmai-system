#!/usr/bin/env python3
"""
Tests for P3T1 - Implement provider_manager.py
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase3.P3T1_Implement_provider_manager import ProviderManager
except ImportError:
    ProviderManager = None

class TestP3T1(unittest.TestCase):
    """Test suite for P3T1 - Implement provider_manager.py"""
    
    def setUp(self):
        if ProviderManager:
            self.component = ProviderManager()
        else:
            self.skipTest("ProviderManager class not found")
    
    def test_component_exists(self):
        """Test that component can be instantiated"""
        if ProviderManager:
            self.assertIsNotNone(self.component)
    
    def test_component_id(self):
        """Test that component has correct ID"""
        if ProviderManager:
            self.assertEqual(self.component.component_id, "P3T1")
    
    def test_create_instance(self):
        """Test instance creation"""
        if ProviderManager:
            instance = self.component.create_instance("aws")
            self.assertIsNotNone(instance)
            self.assertEqual(instance['status'], 'running')

if __name__ == '__main__':
    unittest.main()
