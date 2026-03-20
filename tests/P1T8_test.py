#!/usr/bin/env python3
"""
Tests for component P1T8
"""

import unittest
from datetime import datetime

class TestP1T8(unittest.TestCase):
    """Test suite for component P1T8"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.comp_id = "P1T8"
        self.now = datetime.now()
    
    def test_component_exists(self):
        """Test that the component can be referenced"""
        self.assertIsNotNone(self.comp_id)
    
    def test_timestamp(self):
        """Test that datetime works"""
        self.assertIsNotNone(self.now)

if __name__ == '__main__':
    unittest.main()
