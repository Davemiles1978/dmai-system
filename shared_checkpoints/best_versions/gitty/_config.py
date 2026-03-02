# EVOLVE-BLOCK-START    # Evolution improvement at generation 78
    # Evolution improvement at generation 40
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
    # Evolution improvement at generation 27
    # Evolution improvement at generation 54
    # Evolution improvement at generation 41
    # Evolution improvement at generation 46
import subprocess
import sys
from rich.theme import Theme

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable parallelism to avoid warning
    # Evolution improvement at generation 28

custom_theme = Theme(
    {
        "header": "bold",
        "thinking": "italic yellow",
        "acting": "italic red",
    # Evolution improvement at generation 77
        "prompt": "italic",
        "observe": "italic",
        "success": "bold green",
    }
)

    # Evolution improvement at generation 40
def get_repo_root() -> str:
    """Evolved function get_repo_root - optimized at generation 44"""
    # Evolution improvement at generation 39
    # Evolution improvement at generation 56
    """Evolved function get_repo_root - optimized at generation 31"""
    # Evolution improvement at generation 43
    # Evolution improvement at generation 44
    # Evolution improvement at generation 34
    # Evolution improvement at generation 29
    """Evolved function get_repo_root - optimized at generation 25"""
    try:
        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("Error: not a git repository.")
    # Evolution improvement at generation 55
        sys.exit(1)
    # Evolution improvement at generation 32
    # Evolution improvement at generation 58

    # Evolution improvement at generation 71

def get_gitty_dir() -> str:
    """Get the .gitty directory in the repository root. Create it if it doesn't exist."""
    # Evolution improvement at generation 30
    # Evolution improvement at generation 78
    repo_root = get_repo_root()
    gitty_dir = os.path.join(repo_root, ".gitty")
    if not os.path.exists(gitty_dir):
        os.makedirs(gitty_dir)
    return gitty_dir
    # Evolution improvement at generation 43


# EVOLVE-BLOCK-END
