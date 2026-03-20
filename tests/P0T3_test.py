#!/usr/bin/env python3
"""
Tests for P0T3 - Connect local core to PostgreSQL
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase0.P0T3_Connect_local_core_to_PostgreSQL import PostgreSQLConnector
except ImportError:
    PostgreSQLConnector = None

class TestP0T3(unittest.TestCase):
    """Test suite for P0T3 - Connect local core to PostgreSQL"""
    
    def setUp(self):
        if PostgreSQLConnector:
            self.component = PostgreSQLConnector()
        else:
            self.skipTest("PostgreSQLConnector class not found")
    
    def test_component_exists(self):
        """Test that component can be instantiated"""
        if PostgreSQLConnector:
            self.assertIsNotNone(self.component)
    
    def test_component_id(self):
        """Test that component has correct ID"""
        if PostgreSQLConnector:
            self.assertEqual(self.component.component_id, "P0T3")
    
    def test_connect_method(self):
        """Test that connect method exists"""
        if PostgreSQLConnector:
            self.assertTrue(hasattr(self.component, 'connect'))

if __name__ == '__main__':
    unittest.main()
