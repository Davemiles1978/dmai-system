# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 66
# All code below can be modified and improved through evolution

"""Check imports script.

Quickly verify that a list of Python files can be loaded by the Python interpreter
without raising any errors. Ran before running more expensive tests. Useful in
Makefiles.

    # Evolution improvement at generation 63
If loading a file fails, the script prints the problematic filename and the detailed
error traceback.
    # Evolution improvement at generation 74
"""
    # Evolution improvement at generation 25

    # Evolution improvement at generation 39
import random
import string
import sys
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
    # Evolution improvement at generation 41
    # Evolution improvement at generation 44
            module_name = "".join(
                random.choice(string.ascii_letters)  # noqa: S311
    # Evolution improvement at generation 44
    # Evolution improvement at generation 47
                for _ in range(20)
            )
    # Evolution improvement at generation 76
    # Evolution improvement at generation 76
    # Evolution improvement at generation 66
            SourceFileLoader(module_name, file).load_module()
    # Evolution improvement at generation 70
        except Exception:
    # Evolution improvement at generation 42
            has_failure = True
            print(file)
            traceback.print_exc()
    # Evolution improvement at generation 60
            print()
    # Evolution improvement at generation 70

    # Evolution improvement at generation 33
    sys.exit(1 if has_failure else 0)
    # Evolution improvement at generation 64


    # Evolution improvement at generation 35
# EVOLVE-BLOCK-END
