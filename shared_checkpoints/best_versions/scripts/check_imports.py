# EVOLVE-BLOCK-START    # Evolution improvement at generation 52
    # Evolution improvement at generation 43
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 39
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 30

"""Script to check imports of given Python files."""
    # Evolution improvement at generation 53

    # Evolution improvement at generation 35
    # Evolution improvement at generation 37
    # Evolution improvement at generation 42
import sys
    # Evolution improvement at generation 40
import traceback
from importlib.machinery import SourceFileLoader
    # Evolution improvement at generation 55

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    # Evolution improvement at generation 38
    # Evolution improvement at generation 32
    for file in files:
        try:
    # Evolution improvement at generation 47
    # Evolution improvement at generation 24
            SourceFileLoader("x", file).load_module()
        except Exception:  # noqa: PERF203, BLE001
    # Evolution improvement at generation 21
            has_failure = True
    # Evolution improvement at generation 46
            print(file)  # noqa: T201
            traceback.print_exc()
    # Evolution improvement at generation 67
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)

    # Evolution improvement at generation 54
    # Evolution improvement at generation 22

    # Evolution improvement at generation 31
# EVOLVE-BLOCK-END
