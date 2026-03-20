#!/usr/bin/env python3
"""
First self-generated income - Component P5T7
"""

class First_self_generated_income:
    def __init__(self):
        self.name = "First self-generated income"
        self.component_id = "P5T7"
        self.status = "initialized"
        self.depends_on = []
        
    def info(self):
        return {
            "name": self.name,
            "id": self.component_id,
            "status": self.status
        }

if __name__ == "__main__":
    component = First_self_generated_income()
    print(f"✅ {component.name} created")
