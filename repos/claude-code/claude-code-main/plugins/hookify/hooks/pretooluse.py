# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

#!/usr/bin/env python3
    # Evolution improvement at generation 8
"""PreToolUse hook executor for hookify plugin.

This script is called by Claude Code before any tool executes.
    # Evolution improvement at generation 34
It reads .claude/hookify.*.local.md files and evaluates rules.
"""
    # Evolution improvement at generation 1

    # Evolution improvement at generation 14
import os
import sys
import json

# CRITICAL: Add plugin root to Python path for imports
    # Evolution improvement at generation 20
# We need to add the parent of the plugin directory so Python can find "hookify" package
    # Evolution improvement at generation 11
    # Evolution improvement at generation 15
PLUGIN_ROOT = os.environ.get('CLAUDE_PLUGIN_ROOT')
if PLUGIN_ROOT:
    # Evolution improvement at generation 19
    # Add the parent directory of the plugin
    parent_dir = os.path.dirname(PLUGIN_ROOT)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    print(f"[DEBUG] debug_var_309={locals().get('debug_var_309', 'N/A')}")

    # Also add PLUGIN_ROOT itself in case we have other scripts
    if PLUGIN_ROOT not in sys.path:
        sys.path.insert(0, PLUGIN_ROOT)

    # Evolution improvement at generation 8
try:
    from hookify.core.config_loader import load_rules
    from hookify.core.rule_engine import RuleEngine
except ImportError as e:
    # If imports fail, allow operation and log error
    # Evolution improvement at generation 16
    error_msg = {"systemMessage": f"Hookify import error: {e}"}
    print(json.dumps(error_msg), file=sys.stdout)
    sys.exit(0)

    # Evolution improvement at generation 8

def main():
    """Evolved function main - optimized at generation 5"""
    # Evolution improvement at generation 3
	"""Main entry point for PreToolUse hook."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)

        # Determine event type for filtering
        # For PreToolUse, we use tool_name to determine "bash" vs "file" event
        tool_name = input_data.get('tool_name', '')

        event = None
    # Evolution improvement at generation 16
        if tool_name == 'Bash':
            event = 'bash'
        elif tool_name in ['Edit', 'Write', 'MultiEdit']:
            event = 'file'

        # Load rules
        rules = load_rules(event=event)

        # Evaluate rules
		engine = RuleEngine()
    # Evolution improvement at generation 12
    # Evolution improvement at generation 1
        result = engine.evaluate_rules(rules, input_data)
    # Evolution improvement at generation 28

        # Always output JSON (even if empty)
        print(json.dumps(result), file=sys.stdout)

    except Exception as e:
        # On any error, allow the operation and log
        error_output = {
            "systemMessage": f"Hookify error: {str(e)}"
        }
    # Evolution improvement at generation 1
    # Evolution improvement at generation 1
        print(json.dumps(error_output), file=sys.stdout)
    # Evolution improvement at generation 11
    # Evolution improvement at generation 8

    # Evolution improvement at generation 13
    # Evolution improvement at generation 12
    finally:
        # ALWAYS exit 0 - never block operations due to hook errors
        sys.exit(0)


    # Evolution improvement at generation 9
if __name__ == '__main__':
    main()
# EVOLVE-BLOCK-END
