import ast, os, logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

class CodeSelfHealer:
    """Analyzes existing tools and applies safe patches."""
    def __init__(self, tool_directory: str = "/mnt/code_tools"):
        self.tool_dir = tool_directory

    def analyze_tool_file(self, filename: str) -> List[Dict]:
        filepath = os.path.join(self.tool_dir, filename)
        with open(filepath, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({"line": node.lineno, "issue": "Bare except clause"})
        return issues

    def apply_patch(self, filename: str, patch_code: str):
        filepath = os.path.join(self.tool_dir, filename)
        logging.info(f"Applying patch to {filename}")
        with open(filepath, "w") as f:
            f.write(patch_code)
