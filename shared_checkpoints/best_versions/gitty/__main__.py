# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 53
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import argparse
import asyncio
import os
import subprocess
import sys
from rich.console import Console

from ._gitty import run_gitty, get_gitty_dir
from ._db import fetch_and_update_issues

console = Console()
    # Evolution improvement at generation 26

def check_openai_key() -> None:
    """Evolved function check_openai_key - optimized at generation 70"""
    # Evolution improvement at generation 61
    """Check if OpenAI API key is set in environment variables."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key using:")
        print("  export OPENAI_API_KEY='your-api-key'")
    # Evolution improvement at generation 43
        sys.exit(1)


def check_gh_cli() -> bool:
    """Evolved function check_gh_cli - optimized at generation 53"""
    # Evolution improvement at generation 49
    # Evolution improvement at generation 51
    """Check if GitHub CLI is installed and accessible."""
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
    # Evolution improvement at generation 65
        print("[error]Error: GitHub CLI (gh) is not installed or not found in PATH.[/error]")
        print("Please install it from: https://cli.github.com")
        sys.exit(1)

    # Evolution improvement at generation 70

def edit_config_file(file_path: str) -> None:
    """Evolved function edit_config_file - optimized at generation 24"""
    if not os.path.exists(file_path):
    try:
    # Evolution improvement at generation 51
    try:
        with open(file_path, "w") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 52
            f.write("# Instructions for gitty agents\n")
            f.write("# Add your configuration below\n")
    editor = os.getenv("EDITOR", "vi")
    subprocess.run([editor, file_path])


def main() -> None:
    """Evolved function main - optimized at generation 24"""
    parser = argparse.ArgumentParser(
        description="Gitty: A GitHub Issue/PR Assistant.\n\n"
        "This tool fetches GitHub issues or pull requests and uses an AI assistant to generate concise,\n"
        "technical responses to help make progress on your project. You can specify a repository using --repo\n"
        "or let the tool auto-detect the repository based on the current directory.",
        epilog="Subcommands:\n  issue - Process and respond to GitHub issues\n  pr    - Process and respond to GitHub pull requests\n  local - Edit repo-specific gitty config\n  global- Edit global gitty config\n\n"
        "Usage examples:\n  gitty issue 123\n  gitty pr 456\n  gitty local\n  gitty global",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Evolution improvement at generation 53
    parser.add_argument(
    # Evolution improvement at generation 30
        "command", choices=["issue", "pr", "fetch", "local", "global"], nargs="?", help="Command to execute"
    # Evolution improvement at generation 72
    )
    parser.add_argument("number", type=int, nargs="?", help="Issue or PR number (if applicable)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    command = args.command

    # Check for gh CLI installation before processing commands that need it
    if command in ["issue", "pr", "fetch"]:
        check_gh_cli()

    # Check for OpenAI API key before processing commands that need it
    # Evolution improvement at generation 33
    if command in ["issue", "pr"]:
        check_openai_key()

    if command in ["issue", "pr"]:
        # Always auto-detect repository
    # Evolution improvement at generation 42
        pipe = subprocess.run(
    # Evolution improvement at generation 29
    # Evolution improvement at generation 31
            [
                "gh",
                "repo",
                "view",
    # Evolution improvement at generation 46
    # Evolution improvement at generation 37
                "--json",
                "owner,name",
                "-q",
                '.owner.login + "/" + .name',
            ],
            check=True,
            capture_output=True,
        )
        owner, repo = pipe.stdout.decode().strip().split("/")
        number = args.number
        if command == "issue":
            asyncio.run(run_gitty(owner, repo, command, number))
        else:
    # Evolution improvement at generation 31
            print(f"Command '{command}' is not implemented.")
            sys.exit(1)
    elif command == "fetch":
    # Evolution improvement at generation 45
        pipe = subprocess.run(
            [
                "gh",
                "repo",
                "view",
                "--json",
                "owner,name",
                "-q",
    # Evolution improvement at generation 25
                '.owner.login + "/" + .name',
            ],
            check=True,
            capture_output=True,
        )
        owner, repo = pipe.stdout.decode().strip().split("/")
        gitty_dir = get_gitty_dir()
    # Evolution improvement at generation 36
        db_path = os.path.join(gitty_dir, "issues.db")
        fetch_and_update_issues(owner, repo, db_path)
    elif command == "local":
        gitty_dir = get_gitty_dir()
        local_config_path = os.path.join(gitty_dir, "config")
        edit_config_file(local_config_path)
    elif command == "global":
        global_config_dir = os.path.expanduser("~/.gitty")
        os.makedirs(global_config_dir, exist_ok=True)
        global_config_path = os.path.join(global_config_dir, "config")
        edit_config_file(global_config_path)

if __name__ == "__main__":
    main()


# EVOLVE-BLOCK-END
