#!/usr/bin/env bash
# Install local git hooks. Run once after cloning the repo.
#
# Usage: bash scripts/install-hooks.sh
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"

HOOKS_SRC="scripts/git-hooks"
HOOKS_DST=".git/hooks"

if [[ ! -d .git ]]; then
    echo "Error: not in a git repo (no .git directory at $REPO_ROOT)" >&2
    exit 1
fi

mkdir -p "$HOOKS_DST"

for hook_src in "$HOOKS_SRC"/*; do
    [[ -f "$hook_src" ]] || continue
    hook_name="$(basename "$hook_src")"
    hook_dst="$HOOKS_DST/$hook_name"
    cp "$hook_src" "$hook_dst"
    chmod +x "$hook_dst"
    echo "Installed: $hook_dst"
done

echo ""
echo "Hooks installed. Pushes to refs/heads/main will now run scripts/preflight.sh"
echo "Emergency bypass: git push --no-verify"
