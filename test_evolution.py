#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Automated tests to verify evolution is progressing
"""
import unittest
from pathlib import Path

class TestEvolution(unittest.TestCase):
    def test_checkpoints_exist(self):
        checkpoints = list(Path("shared_checkpoints").glob("gen_*"))
        self.assertGreater(len(checkpoints), 0)
    
    def test_assessments_exist(self):
        assessments = list(Path("shared_data/agi_evolution/assessment").glob("*.json"))
        self.assertGreater(len(assessments), 0)

if __name__ == '__main__':
    unittest.main()
