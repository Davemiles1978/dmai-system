"""GitHubStarMonitor must tolerate a null repo description (PR #164).

GitHub returns ``"description": null`` for repos with no blurb. ``repo.get(
'description', '')`` then returns None (the key exists), and the later
``"AI" in description`` / ``description.lower()`` raised
``argument of type 'NoneType' is not iterable`` every ~1 min. analyze_repo now
coerces description/language to strings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.phase10.GitHubStarMonitor import GitHubStarMonitor


def _monitor(tmp_path):
    return GitHubStarMonitor(data_path=tmp_path, github_username="tester")


def test_analyze_repo_handles_null_description(tmp_path):
    m = _monitor(tmp_path)
    repo = {
        "full_name": "acme/thing",
        "description": None,  # GitHub null
        "language": None,
        "html_url": "https://github.com/acme/thing",
        "clone_url": "https://github.com/acme/thing.git",
        "stargazers_count": 3,
    }
    analysis = m.analyze_repo(repo)  # must not raise
    assert analysis["name"] == "acme/thing"
    assert analysis["type"] == "reference"
    assert analysis["language"] == "Unknown"


def test_analyze_repo_ai_keyword_still_matches(tmp_path):
    m = _monitor(tmp_path)
    repo = {
        "full_name": "acme/ai-lib",
        "description": "An AI toolkit",
        "language": "Rust",
        "html_url": "u",
        "clone_url": "c",
        "stargazers_count": 1,
    }
    analysis = m.analyze_repo(repo)
    assert analysis["type"] == "ai_tool"
