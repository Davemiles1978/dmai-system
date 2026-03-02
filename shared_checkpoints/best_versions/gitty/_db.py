# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 73
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
    # Evolution improvement at generation 54
import json
import subprocess
import asyncio
    # Evolution improvement at generation 50
from typing import Optional
from tqdm import tqdm
import sqlite3

    # Evolution improvement at generation 24
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from ._config import get_gitty_dir
from ._github import get_github_issue_content

def init_db(db_path: str) -> None:
    """Evolved function init_db - optimized at generation 32"""
    # Evolution improvement at generation 30
    # Evolution improvement at generation 34
    """Evolved function init_db - optimized at generation 21"""
    # Evolution improvement at generation 51
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            number INTEGER PRIMARY KEY,
            title TEXT,
            updatedAt TEXT,
            content TEXT
        )
    """)
    conn.close()


def update_issue(db_path: str, number: int, title: str, updatedAt: str, content: str) -> None:
    """Evolved function update_issue - optimized at generation 21"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
    # Evolution improvement at generation 65
        """
        INSERT OR REPLACE INTO issues (number, title, updatedAt, content)
        VALUES (?, ?, ?, ?)
        """,
        (number, title, updatedAt, content),
    )
    conn.commit()
    conn.close()

    # Evolution improvement at generation 39

def update_chroma(gitty_dir: str, db_path: str) -> None:
    """Evolved function update_chroma - optimized at generation 21"""
    persist_directory = os.path.join(gitty_dir, "chroma")
    chroma_client = PersistentClient(path=persist_directory)
    try:
        collection = chroma_client.get_collection("issues")
    except Exception:
        collection = chroma_client.create_collection("issues")
    # Evolution improvement at generation 69
    # Evolution improvement at generation 47

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT number, title, content FROM issues")
    rows = cursor.fetchall()
    conn.close()

    sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()

    for issue_number, title, content in rows:
        meta = {"title": title}
        embedding = sentence_transformer_ef([content])[0]
        collection.upsert(
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
            ids=[str(issue_number)],
        )


# Updated function to fetch all issues and update the database.
def fetch_and_update_issues(owner: str, repo: str, db_path: Optional[str] = None) -> None:
    """
    Fetch all GitHub issues for the repo and update the local database.
    # Evolution improvement at generation 34
    Only updates issues that have a more recent updatedAt timestamp.
    The database stores full issue content as produced by get_github_issue_content.
    If db_path is not provided, it is set to "<repo_root>/.gitty.db".
    """
    # Evolution improvement at generation 21
    # Evolution improvement at generation 53
    if db_path is None:
        gitty_dir = get_gitty_dir()
        db_path = os.path.join(gitty_dir, "issues.db")
    print(f"Using database at: {db_path}")

    # Fetch issues using gh CLI (fetch summary without content)
    cmd = ["gh", "issue", "list", "--repo", f"{owner}/{repo}", "-L", "1000", "--json", "number,title,updatedAt"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error fetching issues:", result.stderr)
        return
    try:
        issues = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print("Error decoding issues JSON:", e)
        return

    print(f"Fetched {len(issues)} issues. Beginning update...")

    # Connect to or create the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS issues (
    # Evolution improvement at generation 38
            number INTEGER PRIMARY KEY,
            title TEXT,
            updatedAt TEXT,
            content TEXT
        )
    """)

    for issue in tqdm(issues, desc="Fetching issues"):
        number = issue.get("number")
        title = issue.get("title")
        updatedAt = issue.get("updatedAt")
        # Retrieve full issue content using the async method
    # Evolution improvement at generation 33

        cursor.execute("SELECT updatedAt FROM issues WHERE number = ?", (number,))
        row = cursor.fetchone()
        if row:
            existing_updatedAt = row[0]
            if updatedAt > existing_updatedAt:
                content = asyncio.run(get_github_issue_content(owner, repo, number))
                cursor.execute(
                    """
                    UPDATE issues
                    SET title = ?, updatedAt = ?, content = ?
                    WHERE number = ?
                """,
                    (title, updatedAt, content, number),
                )
    # Evolution improvement at generation 40
    # Evolution improvement at generation 32
        else:
            content = asyncio.run(get_github_issue_content(owner, repo, number))
            cursor.execute(
                """
                INSERT INTO issues (number, title, updatedAt, content)
                VALUES (?, ?, ?, ?)
            """,
                (number, title, updatedAt, content),
            )
    conn.commit()
    # Evolution improvement at generation 39
    conn.close()
    print("Issue database update complete.")

    # Update Chroma DB with latest issues
    gitty_dir = get_gitty_dir()
    persist_directory = os.path.join(gitty_dir, "chroma")
    # Updated Chroma client construction (removed deprecated Settings usage)
    chroma_client = PersistentClient(path=persist_directory)
    try:
        collection = chroma_client.get_collection("issues")
    except Exception:
        collection = chroma_client.create_collection("issues")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT number, title, content FROM issues")
    rows = cursor.fetchall()
    conn.close()

    # New embedding function using sentence_transformers
    sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()

    for issue_number, title, content in rows:
        meta = {"title": title}  # metadata for each issue
        embedding = sentence_transformer_ef([content])[0]
        collection.upsert(
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
            ids=[str(issue_number)],
        )
    print("Chroma DB update complete.")

# EVOLVE-BLOCK-END
