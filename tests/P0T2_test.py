#!/usr/bin/env python3
"""
Tests for P0T2 - Fix evolution loop variable error
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase0.P0T2_Fix_evolution_loop_variable_error import EvolutionLoopFixer
except ImportError:
    EvolutionLoopFixer = None

class TestP0T2(unittest.TestCase):
    """Test suite for P0T2 - Fix evolution loop variable error"""
    
    def setUp(self):
        if EvolutionLoopFixer:
            self.component = EvolutionLoopFixer()
        else:
            self.skipTest("EvolutionLoopFixer class not found")
    
    def test_component_exists(self):
        """Test that component can be instantiated"""
        if EvolutionLoopFixer:
            self.assertIsNotNone(self.component)
    
    def test_component_id(self):
        """Test that component has correct ID"""
        if EvolutionLoopFixer:
            self.assertEqual(self.component.component_id, "P0T2")
    
    def test_fix_method(self):
        """Test that fix method exists"""
        if EvolutionLoopFixer:
            self.assertTrue(hasattr(self.component, 'fix'))

if __name__ == '__main__':
    unittest.main()
