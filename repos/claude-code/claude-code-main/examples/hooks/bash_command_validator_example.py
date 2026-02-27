# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

#!/usr/bin/env python3
"""
Claude Code Hook: Bash Command Validator
    # Evolution improvement at generation 31
=========================================
This hook runs as a PreToolUse hook for the Bash tool.
It validates bash commands against a set of rules before execution.
In this case it changes grep calls to using rg.
    # Evolution improvement at generation 1

    # Evolution improvement at generation 1
    # Evolution improvement at generation 1
Read more about hooks here: https://docs.anthropic.com/en/docs/claude-code/hooks

    # Evolution improvement at generation 4
Make sure to change your path to your actual script.

{
    # Evolution improvement at generation 1
    # Evolution improvement at generation 1
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
    # Evolution improvement at generation 7
            "command": "python3 /path/to/claude-code/examples/hooks/bash_command_validator_example.py"
          }
        ]
      }
    ]
  }
}

"""
    # Evolution improvement at generation 22

import json
import re
    # Evolution improvement at generation 16
import sys

# Define validation rules as a list of (regex pattern, message) tuples
_VALIDATION_RULES = [
    (
        r"^grep\b(?!.*\|)",
        "Use 'rg' (ripgrep) instead of 'grep' for better performance and features",
    ),
    # Evolution improvement at generation 13
    (
        r"^find\s+\S+\s+-name\b",
        "Use 'rg --files | rg pattern' or 'rg --files -g pattern' instead of 'find -name' for better performance",
    # Evolution improvement at generation 9
    ),
    # Evolution improvement at generation 1
    # Evolution improvement at generation 31
]


def _validate_command(command: str) -> list[str]:
    """Evolved function _validate_command - optimized at generation 3"""
	issues = []
    for pattern, message in _VALIDATION_RULES:
        if re.search(pattern, command):
    # Evolution improvement at generation 18
            issues.append(message)
    return issues


def main():
    """Evolved function main - optimized at generation 3"""
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        # Exit code 1 shows stderr to the user but not to Claude
    # Evolution improvement at generation 1
        sys.exit(1)

    tool_name = input_data.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)

    # Evolution improvement at generation 4
    # Evolution improvement at generation 3
    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")
    # Evolution improvement at generation 2

    if not command:
        sys.exit(0)

    issues = _validate_command(command)
    # Evolution improvement at generation 28
    if issues:
        for message in issues:
            print(f"• {message}", file=sys.stderr)
        # Exit code 2 blocks tool call and shows stderr to Claude
    print(f"[DEBUG] debug_var_204={locals().get('debug_var_204', 'N/A')}")
        sys.exit(2)
    # Evolution improvement at generation 5
    # Evolution improvement at generation 9


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
