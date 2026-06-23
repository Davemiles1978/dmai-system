#!/usr/bin/env python3
"""
DMAI Graph Evolution Monitor
Reads discoveries.jsonl and insights.jsonl, diffs against graph_schema.json,
and autonomously grows the knowledge graph with new neurons and synapses.

Run manually:  python scripts/graph_evolution_monitor.py
Friday cron:   Reads discoveries, updates schema, commits to auto/graph-update-YYYY-MM-DD branch
"""

import json
import os
import sys
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
SCHEMA_PATH    = REPO_ROOT / "aevora-training" / "dashboard" / "data" / "graph_schema.json"
DISCOVERIES    = REPO_ROOT / "data" / "research" / "discoveries.jsonl"
INSIGHTS       = REPO_ROOT / "data" / "research" / "insights.jsonl"
CHANGELOG_DIR  = REPO_ROOT / "data" / "graph_evolution"
CHANGELOG_FILE = CHANGELOG_DIR / "changelog.jsonl"

# ── Domain → cluster mapping ───────────────────────────────────────────────────
DOMAIN_CLUSTER_MAP = {
    "machine_learning":    "learning",
    "reinforcement_learning": "learning",
    "autonomous_agents":   "research",
    "trading":             "revenue",
    "content_generation":  "revenue",
    "computer_vision":     "research",
    "nlp":                 "knowledge",
    "self_improvement":    "core",
    "knowledge_systems":   "knowledge",
    "robotics":            "research",
    "cybersecurity":       "research",
    "web_technologies":    "research",
    "data_science":        "knowledge",
    "cloud_devops":        "providers",
}

CLUSTER_COLORS = {
    "core":      "#6c63ff",
    "learning":  "#00d4aa",
    "research":  "#ffa502",
    "knowledge": "#a29bfe",
    "providers": "#74b9ff",
    "revenue":   "#ff4757",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_schema() -> dict:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"graph_schema.json not found at {SCHEMA_PATH}")
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def save_schema(schema: dict) -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCHEMA_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(schema, f, indent=2)
    tmp.replace(SCHEMA_PATH)


def existing_node_ids(schema: dict) -> set:
    return {n["id"] for n in schema.get("neurons", [])}


def existing_synapse_keys(schema: dict) -> set:
    return {(s["source"], s["target"]) for s in schema.get("synapses", [])}


def slug(text: str) -> str:
    """Convert a label/domain to a safe node ID slug."""
    import re
    return re.sub(r"[^a-z0-9_]", "_", text.lower().strip())[:48]


def stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha1(text.encode()).hexdigest()[:6]
    return f"{slug(prefix)}_{h}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_changelog(entry: dict) -> None:
    CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHANGELOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Core evolution logic ───────────────────────────────────────────────────────

def evolve(schema: dict, discoveries: list[dict], insights: list[dict]) -> tuple[dict, list[str]]:
    """
    Returns (updated_schema, list_of_change_descriptions).
    Mutates a deep copy of schema.
    """
    import copy
    schema = copy.deepcopy(schema)
    changes = []
    existing_ids = existing_node_ids(schema)
    existing_synapses = existing_synapse_keys(schema)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Process discoveries from autonomous_researcher ──────────────────────
    for disc in discoveries:
        domain   = disc.get("domain", "")
        entities = disc.get("entities", [])
        source   = disc.get("source", "")
        date_str = disc.get("date", now[:10])

        # 1. Ensure domain node exists
        if domain and domain in DOMAIN_CLUSTER_MAP:
            domain_id = slug(domain)
            if domain_id not in existing_ids:
                cluster = DOMAIN_CLUSTER_MAP[domain]
                node = {
                    "id": domain_id,
                    "label": domain.replace("_", " ").title(),
                    "cluster": cluster,
                    "description": f"Auto-discovered domain: {domain}",
                    "activation": 0.6,
                    "auto_generated": True,
                    "first_seen": date_str,
                    "source": source,
                }
                schema["neurons"].append(node)
                existing_ids.add(domain_id)
                # Wire domain node to its cluster hub
                hub_map = {
                    "core": "dmai_core", "learning": "learning_orch",
                    "research": "auto_researcher", "knowledge": "knowledge_mgr",
                    "providers": "ai_hub", "revenue": "self_funding",
                }
                hub = hub_map.get(cluster)
                if hub and hub in existing_ids:
                    key = (hub, domain_id)
                    if key not in existing_synapses:
                        schema["synapses"].append({"source": hub, "target": domain_id, "weight": 0.6, "type": "data", "auto_generated": True})
                        existing_synapses.add(key)
                changes.append(f"NEW NEURON: {domain_id} (cluster={cluster}, source={source})")
                append_changelog({"ts": now, "type": "new_neuron", "id": domain_id, "cluster": cluster, "source": source})

        # 2. Ensure entity nodes exist and connect them to their domain
        for entity in entities[:5]:  # cap at 5 entities per discovery cycle
            entity_id = stable_id(domain or "entity", entity)
            if entity_id not in existing_ids:
                cluster = DOMAIN_CLUSTER_MAP.get(domain, "knowledge")
                node = {
                    "id": entity_id,
                    "label": entity[:32],
                    "cluster": cluster,
                    "description": f"Auto-discovered entity from {source}",
                    "activation": 0.5,
                    "auto_generated": True,
                    "first_seen": date_str,
                    "source": source,
                }
                schema["neurons"].append(node)
                existing_ids.add(entity_id)
                # Connect entity → domain node (if domain node exists)
                domain_id = slug(domain) if domain else None
                if domain_id and domain_id in existing_ids:
                    key = (domain_id, entity_id)
                    if key not in existing_synapses:
                        schema["synapses"].append({"source": domain_id, "target": entity_id, "weight": 0.5, "type": "data", "auto_generated": True})
                        existing_synapses.add(key)
                changes.append(f"NEW ENTITY NEURON: {entity_id} label='{entity[:32]}' (source={source})")
                append_changelog({"ts": now, "type": "new_entity", "id": entity_id, "label": entity[:32], "source": source})

    # ── Process insights from si_core ────────────────────────────────────────
    for ins in insights:
        domain  = ins.get("domain", "")
        concept = ins.get("concept", ins.get("insight", ""))
        source  = ins.get("source", "si_core")
        if not concept:
            continue
        node_id = stable_id("insight", concept)
        if node_id not in existing_ids:
            cluster = DOMAIN_CLUSTER_MAP.get(domain, "knowledge")
            node = {
                "id": node_id,
                "label": concept[:32],
                "cluster": cluster,
                "description": f"Insight from {source}: {concept[:80]}",
                "activation": 0.55,
                "auto_generated": True,
                "first_seen": ins.get("date", now[:10]),
                "source": source,
            }
            schema["neurons"].append(node)
            existing_ids.add(node_id)
            # Connect insight → si_core feedback
            key = (node_id, "si_core")
            if key not in existing_synapses and "si_core" in existing_ids:
                schema["synapses"].append({"source": node_id, "target": "si_core", "weight": 0.55, "type": "feedback", "auto_generated": True})
                existing_synapses.add(key)
            changes.append(f"NEW INSIGHT NEURON: {node_id} concept='{concept[:32]}'")
            append_changelog({"ts": now, "type": "new_insight", "id": node_id, "concept": concept[:32], "source": source})

    # ── Update schema metadata ───────────────────────────────────────────────
    if changes:
        schema["total_neurons"]  = len(schema["neurons"])
        schema["total_synapses"] = len(schema["synapses"])
        schema["evolution_cycle"] = schema.get("evolution_cycle", 0) + 1
        schema["last_updated"]   = now[:10]
        schema["metadata"]["auto_evolved"] = True

    return schema, changes


# ── Git helpers ────────────────────────────────────────────────────────────────

def git_run(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=check, capture_output=True, text=True)


def create_pr_branch_and_push(changes: list[str]) -> str | None:
    """Create a dated branch, commit graph_schema.json + changelog, push, return branch name."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    branch = f"auto/graph-update-{today}"

    # Ensure we're on main and up-to-date
    git_run("checkout", "main")
    git_run("pull", "--ff-only", "origin", "main", check=False)

    # Create branch
    git_run("checkout", "-b", branch)

    # Stage files
    files_to_add = [str(SCHEMA_PATH.relative_to(REPO_ROOT))]
    if CHANGELOG_FILE.exists():
        files_to_add.append(str(CHANGELOG_FILE.relative_to(REPO_ROOT)))

    git_run("add", *files_to_add)

    commit_msg = (
        f"[Auto] Knowledge Graph Evolution — {today}\n\n"
        f"Changes ({len(changes)}):\n" +
        "\n".join(f"  - {c}" for c in changes[:30])
    )
    git_run("commit", "-m", commit_msg)
    git_run("push", "-u", "origin", branch)

    return branch


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DMAI Graph Evolution Monitor")
    parser.add_argument("--dry-run",   action="store_true", help="Print changes without writing")
    parser.add_argument("--no-branch", action="store_true", help="Update schema locally without creating a git branch")
    args = parser.parse_args()

    print(f"[graph_evolution_monitor] Starting — {datetime.now(timezone.utc).isoformat()}")

    schema      = load_schema()
    discoveries = load_jsonl(DISCOVERIES)
    insights    = load_jsonl(INSIGHTS)

    print(f"  Loaded schema v{schema.get('schema_version')} — "
          f"{schema['total_neurons']} neurons, {schema['total_synapses']} synapses")
    print(f"  Processing {len(discoveries)} discoveries, {len(insights)} insights")

    updated_schema, changes = evolve(schema, discoveries, insights)

    if not changes:
        print("  No new neurons or synapses detected — schema is current.")
        sys.exit(0)

    print(f"  {len(changes)} change(s) detected:")
    for c in changes:
        print(f"    + {c}")

    if args.dry_run:
        print("  [dry-run] No files written.")
        sys.exit(0)

    save_schema(updated_schema)
    print(f"  Schema updated → {SCHEMA_PATH}")

    if not args.no_branch:
        branch = create_pr_branch_and_push(changes)
        if branch:
            print(f"  Branch pushed: {branch}")
            print(f"  Open PR at: https://github.com/Davemiles1978/dmai-system/compare/{branch}")
    else:
        print("  [no-branch] Skipped git operations.")


if __name__ == "__main__":
    main()
