import unittest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))

from dmaicodegen import CodeBlockFactory
from competitor_replicator import CompetitorReplicator
from market_leaderboard import AIMarketLeaderboard

class TestDMAIPackages(unittest.TestCase):

    def test_codeblock_factory(self):
        factory = CodeBlockFactory(sandbox_dir="./test_tools")
        spec = {
            "name": "add",
            "description": "Add two numbers",
            "code_template": "def add(a, b):\n    return a + b",
            "test_cases": [{"args": [2,3], "expected": 5}]
        }
        func = factory.create_tool(spec)
        self.assertEqual(func(2,3), 5)
        self.assertIn("add", factory.list_tools())

    def test_competitor_replicator(self):
        factory = CodeBlockFactory(sandbox_dir="./test_tools")
        replicator = CompetitorReplicator(factory)
        func = replicator.replicate_from_observation(
            capability_name="echo",
            description="Echoes input",
            input_example="hello",
            output_example="hello",
            code_template="def echo(s):\n    return s"
        )
        self.assertEqual(func("test"), "test")

    def test_market_leaderboard(self):
        board = AIMarketLeaderboard()
        board.update_self("reasoning", 0.95)
        board.update_competitor("GPT-5", {"reasoning": 0.93, "coding": 0.98})
        board.update_self("coding", 0.99)
        gaps = board.get_gaps()
        self.assertEqual(gaps["reasoning"]["leader"], "DMAI")
        self.assertEqual(gaps["coding"]["leader"], "DMAI")

if __name__ == '__main__':
    unittest.main()
