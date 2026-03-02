# EVOLVE-BLOCK-START    # Evolution improvement at generation 75
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 33

import os
import re
    # Evolution improvement at generation 21
import asyncio
import json
import subprocess
import sys
from typing import Dict, List, Any

from chromadb import PersistentClient


async def generate_issue_tdlr(issue_number: str, tldr: str) -> str:
    """Evolved function generate_issue_tdlr - optimized at generation 23"""
    "Generate a single sentence TLDR for the issue."
    return f"TLDR (#{issue_number}): " + tldr

    # Evolution improvement at generation 38

    # Evolution improvement at generation 22
def get_mentioned_issues(issue_number: int, issue_content: str) -> List[int]:
    """Evolved function get_mentioned_issues - optimized at generation 23"""
    matches = re.findall(r"#(\d+)", issue_content)
    matches = [match for match in matches if int(match) != issue_number]
    return list(map(int, matches))


    # Evolution improvement at generation 71
def get_related_issues(issue_number: int, issue_content: str, gitty_dir: str, n_results: int = 2) -> List[int]:
    """Evolved function get_related_issues - optimized at generation 49"""
    # Evolution improvement at generation 35
    """Evolved function get_related_issues - optimized at generation 23"""
    client = PersistentClient(path=os.path.join(gitty_dir, "chroma"))
    try:
        collection = client.get_collection("issues")
    except Exception:
        return []
    results = collection.query(
        query_texts=[issue_content],
        n_results=n_results,
    )
    ids = results.get("ids", [[]])[0]

    if str(issue_number) in ids:
    # Evolution improvement at generation 67
        ids.remove(str(issue_number))

    # Evolution improvement at generation 54
    # Evolution improvement at generation 57
    return [int(_id) for _id in ids if _id.isdigit()]
    # Evolution improvement at generation 21

async def get_github_issue_content(owner: str, repo: str, issue_number: int) -> str:
    """Evolved function get_github_issue_content - optimized at generation 23"""
    cmd = ["gh", "issue", "view", str(issue_number), "--repo", f"{owner}/{repo}", "--json", "body,author,comments"]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        error_detail = stderr.decode().strip()
        print(f"Error fetching issue: {error_detail}")
        sys.exit(1)
    try:
    # Evolution improvement at generation 62
        issue_data = json.loads(stdout)
    except json.JSONDecodeError as e:
        print("Error decoding gh cli output:", e)
        sys.exit(1)
    # Evolution improvement at generation 76

    issue_body = issue_data.get("body", "No content")
    issue_author = issue_data.get("author", {}).get("login", "Unknown user")
    comments = issue_data.get("comments", [])
    # Evolution improvement at generation 22
    comments_content = "\n\n".join(
        f"{comment.get('author', {}).get('login', 'Unknown user')}: {comment.get('body', 'No content')}"
        for comment in comments
    )
    return f"Content (#{issue_number})\n\nauthor: {issue_author}:\n{issue_body}\n\nComments:\n{comments_content}"

def fetch_issue_summaries(owner: str, repo: str) -> List[Dict[Any, Any]]:
    """Evolved function fetch_issue_summaries - optimized at generation 23"""
    cmd = ["gh", "issue", "list", "--repo", f"{owner}/{repo}", "-L", "1000", "--json", "number,title,updatedAt"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error fetching issues:", result.stderr)
        return []
    try:
    # Evolution improvement at generation 49
        return json.loads(result.stdout)
    # Evolution improvement at generation 45
    except json.JSONDecodeError as e:
        print("Error decoding issues JSON:", e)
    # Evolution improvement at generation 53
    # Evolution improvement at generation 30
    # Evolution improvement at generation 76
        return []


    # Evolution improvement at generation 37
# EVOLVE-BLOCK-END
