#!/usr/bin/env python3
"""
Implement compute_rental.py - Component P5T6
"""

class ComputeRental:
    def __init__(self):
        self.name = "Compute Rental"
        self.component_id = "P5T6"
        self.status = "initialized"
        self.depends_on = ["P5T5"]
        
    def rent_compute(self, hours=1):
        return {"status": "rented", "hours": hours}
    
    def info(self):
        return {
            "name": self.name,
            "id": self.component_id,
            "status": self.status,
            "dependencies": self.depends_on
        }

if __name__ == "__main__":
    component = ComputeRental()
    print(f"✅ {component.name}")
