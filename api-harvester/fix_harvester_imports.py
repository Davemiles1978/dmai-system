#!/usr/bin/env python3
"""
Fix harvester imports for evolution
"""
import re

with open('harvester.py', 'r') as f:
    content = f.read()

# Add imports after the last import
import_section = re.search(r'^(import .*|from .* import .*)$', content, re.MULTILINE)
if import_section:
    # Find where to insert
    lines = content.split('\n')
    last_import = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import = i
    
    # Insert evolution imports
    lines.insert(last_import + 1, 'from db_integration import KeyEvolutionDB, process_harvested_key')
    lines.insert(last_import + 2, 'from github_scraper_evolution import GitHubScraperEvolution')
    content = '\n'.join(lines)
    print("✅ Added evolution imports")

# Replace GitHubScraper with GitHubScraperEvolution
content = content.replace('GitHubScraper(config=github_config)', 'GitHubScraperEvolution(config=github_config)')
print("✅ Replaced with evolution scraper")

# Write the fixed file
with open('harvester.py', 'w') as f:
    f.write(content)

print("✅ Harvester imports fixed!")
