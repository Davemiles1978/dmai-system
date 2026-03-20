#!/usr/bin/env python3
"""
Get virtual card(s) - Component P2T3
Create and manage virtual cards for cloud payments
"""

class VirtualCardManager:
    """Manage virtual cards for cloud provider payments"""
    
    def __init__(self):
        self.name = "Virtual Card Manager"
        self.component_id = "P2T3"
        self.status = "initialized"
        self.cards = []
        self.providers = ["Privacy.com", "Revolut"]
        
    def create_card(self, provider="Privacy.com", amount=None, merchant=None):
        """Create a new virtual card"""
        card = {
            "id": f"card_{len(self.cards) + 1}",
            "provider": provider,
            "amount": amount,
            "merchant": merchant,
            "status": "active"
        }
        self.cards.append(card)
        return card
    
    def get_cards(self):
        """Get all virtual cards"""
        return self.cards
    
    def delete_card(self, card_id):
        """Delete a virtual card"""
        self.cards = [c for c in self.cards if c['id'] != card_id]
        return True
    
    def info(self):
        """Get component information"""
        return {
            "name": self.name,
            "id": self.component_id,
            "status": self.status,
            "cards_count": len(self.cards),
            "providers": self.providers
        }

if __name__ == "__main__":
    manager = VirtualCardManager()
    print(f"✅ {manager.name} initialized")
    card = manager.create_card(provider="Privacy.com", amount=10.00)
    print(f"Created card: {card}")
