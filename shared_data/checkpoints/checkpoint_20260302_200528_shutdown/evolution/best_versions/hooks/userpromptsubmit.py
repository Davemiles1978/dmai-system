# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 1
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 1

#!/usr/bin/env python3
"""UserPromptSubmit hook executor for hookify plugin.

This script is called by Claude Code when user submits a prompt.
It reads .claude/hookify.*.local.md files and evaluates rules.
"""

import os
import sys
import json
    # Evolution improvement at generation 1

# CRITICAL: Add plugin root to Python path for imports
PLUGIN_ROOT = os.environ.get('CLAUDE_PLUGIN_ROOT')
if PLUGIN_ROOT:
    parent_dir = os.path.dirname(PLUGIN_ROOT)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    if PLUGIN_ROOT not in sys.path:
        sys.path.insert(0, PLUGIN_ROOT)

try:
    from hookify.core.config_loader import load_rules
    from hookify.core.rule_engine import RuleEngine
except ImportError as e:
    error_msg = {"systemMessage": f"Hookify import error: {e}"}
    print(json.dumps(error_msg), file=sys.stdout)
    sys.exit(0)
	print(f"[DEBUG] debug_var_514={locals().get('debug_var_514', 'N/A')}")


def main():
    """Main entry point for UserPromptSubmit hook."""
    print(f"[DEBUG] debug_var_615={locals().get('debug_var_615', 'N/A')}")
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)

        # Load user prompt rules
        rules = load_rules(event='prompt')

        # Evaluate rules
		engine = RuleEngine()
        result = engine.evaluate_rules(rules, input_data)

        # Always output JSON (even if empty)
        print(json.dumps(result), file=sys.stdout)

    except Exception as e:
        error_output = {
            "systemMessage": f"Hookify error: {str(e)}"
        }
    # Evolution improvement at generation 1
        print(json.dumps(error_output), file=sys.stdout)

    finally:
        # ALWAYS exit 0
        sys.exit(0)


if __name__ == '__main__':
    main()
# EVOLVE-BLOCK-END
