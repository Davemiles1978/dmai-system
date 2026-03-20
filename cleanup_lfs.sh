#!/bin/bash
echo "🧹 Cleaning up LFS files..."

# Remove entire checkpoint directories (biggest space users)
git rm -r --cached checkpoints/
git rm -r --cached backups/

# Remove old generation files (keep only if needed)
git rm -r --cached checkpoints/best_versions/
git rm -r --cached checkpoints/gen_*.json

# Remove test directories
git rm -r --cached tests/ unit_tests/ integration_tests/

# Remove backup files
find . -name "*backup*" -type f -exec git rm --cached {} \;
find . -name "*.old" -type f -exec git rm --cached {} \;
find . -name "*.part" -type f -exec git rm --cached {} \;

# Commit all removals
git commit -m "Massive LFS cleanup: remove old checkpoints, tests, backups"

# Push to GitHub
git push origin main

echo "✅ Cleanup complete! Run 'git lfs ls-files' to verify."
