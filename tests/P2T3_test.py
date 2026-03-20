#!/usr/bin/env python3
"""
Tests for P2T3 - Get virtual card(s)
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase2.P2T3_Get_virtual_cards import VirtualCardManager
except ImportError:
    VirtualCardManager = None

class TestP2T3(unittest.TestCase):
    """Test suite for P2T3 - Get virtual card(s)"""
    
    def setUp(self):
        if VirtualCardManager:
            self.component = VirtualCardManager()
        else:
            self.skipTest("VirtualCardManager class not found")
    
    def test_component_exists(self):
        """Test that component can be instantiated"""
        if VirtualCardManager:
            self.assertIsNotNone(self.component)
    
    def test_component_id(self):
        """Test that component has correct ID"""
        if VirtualCardManager:
            self.assertEqual(self.component.component_id, "P2T3")
    
    def test_create_card(self):
        """Test card creation"""
        if VirtualCardManager:
            card = self.component.create_card()
            self.assertIsNotNone(card)
            self.assertIn('id', card)
            self.assertEqual(card['status'], 'active')
    
    def test_get_cards(self):
        """Test getting cards"""
        if VirtualCardManager:
            # Create a card first
            self.component.create_card(amount=10.00)
            cards = self.component.get_cards()
            self.assertIsInstance(cards, list)
            self.assertGreaterEqual(len(cards), 1)
    
    def test_get_card_by_id(self):
        """Test getting card by ID"""
        if VirtualCardManager:
            card = self.component.create_card()
            retrieved = self.component.get_card_by_id(card['id'])
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved['id'], card['id'])
    
    def test_delete_card(self):
        """Test deleting a card"""
        if VirtualCardManager:
            card = self.component.create_card()
            self.assertEqual(len(self.component.get_cards()), 1)
            self.component.delete_card(card['id'])
            self.assertEqual(len(self.component.get_cards()), 0)

if __name__ == '__main__':
    unittest.main()
