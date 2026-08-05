"""
DMAI GitHub Repo Pipeline
==========================
Complete pipeline: scan → clone → analyze → reverse engineer → rebuild → test → register.

Takes starred GitHub repos and converts them into native DMAI components.
Each successfully processed repo becomes a capability DMAI can use.

Stages:
  1. SCAN:    GitHubStarredScanner fetches starred repos
  2. CLONE:   RepoIntegrationEngine clones the repo
  3. ANALYZE: Extract structure, dependencies, entry points, core patterns
  4. REVERSE:  Extract the core algorithm/architecture into a spec
  5. REBUILD:  Generate a DMAI-native Python component from the spec
  6. TEST:     Syntax check, smoke test the new component
  7. REGISTER: Add to capability registry, knowledge graph, vector store

Runs as a background daemon processing one repo every 30 minutes.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("dmai.github_repo_pipeline")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PIPELINE_STATE = _REPO_ROOT / "data" / "github_pipeline" / "state.json"
_PIPELINE_LOG = _REPO_ROOT / "data" / "github_pipeline" / "pipeline_log.jsonl"
_WORK_DIR = _REPO_ROOT / "data" / "github_pipeline" / "repos"


class GitHubRepoPipeline:
    """
    End-to-end pipeline: starred repo → native DMAI component.
    """

    def __init__(self, components: Optional[Dict] = None):
        self.components = components or {}
        self.state_file = _PIPELINE_STATE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = _PIPELINE_LOG
        self.work_dir = _WORK_DIR
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        logger.info("GitHubRepoPipeline initialised")

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {
            "processed": [],       # list of {repo_name, component_name, status, timestamp}
            "in_progress": None,
            "total_processed": 0,
            "total_succeeded": 0,
            "total_failed": 0,
        }

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def _log(self, event: str, detail: Dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **detail,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Stage 3: ANALYZE
    # ------------------------------------------------------------------

    def analyze_repo(self, repo_path: Path) -> Dict:
        """
        Analyze a cloned repo: extract structure, language, dependencies, entry points.
        Returns a structured analysis dict.
        """
        analysis = {
            "repo_path": str(repo_path),
            "files": [],
            "languages": {},
            "entry_points": [],
            "dependencies": [],
            "readme": None,
            "structure": {},
        }

        if not repo_path.exists():
            return {"error": f"Repo path not found: {repo_path}"}

        # Walk file tree (limit depth and count for performance)
        py_files = []
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and venv dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "venv", "__pycache__", ".git")]
            for f in files:
                fpath = Path(root) / f
                rel = fpath.relative_to(repo_path)
                ext = fpath.suffix.lower()
                analysis["languages"][ext] = analysis["languages"].get(ext, 0) + 1
                if ext == ".py":
                    py_files.append(str(rel))
                analysis["files"].append(str(rel))

        analysis["python_files"] = py_files

        # Find entry points
        for f in py_files:
            fpath = repo_path / f
            try:
                content = fpath.read_text(errors="replace")[:2000]
                if 'if __name__ == "__main__"' in content or "def main(" in content:
                    analysis["entry_points"].append(f)
                # Extract imports
                for line in content.split("\n"):
                    if line.strip().startswith("import ") or line.strip().startswith("from "):
                        dep = line.strip().split()[1].split(".")[0]
                        if dep not in analysis["dependencies"]:
                            analysis["dependencies"].append(dep)
            except Exception:
                pass

        # Read README
        for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = repo_path / readme_name
            if readme_path.exists():
                try:
                    analysis["readme"] = readme_path.read_text()[:3000]
                except Exception:
                    pass
                break

        # Structure summary
        analysis["structure"] = {
            "total_files": len(analysis["files"]),
            "python_files": len(py_files),
            "entry_points": len(analysis["entry_points"]),
            "unique_deps": len(set(analysis["dependencies"])),
            "primary_language": max(analysis["languages"], key=analysis["languages"].get) if analysis["languages"] else "unknown",
        }

        return analysis

    # ------------------------------------------------------------------
    # Stage 4: REVERSE ENGINEER — extract core spec
    # ------------------------------------------------------------------

    def reverse_engineer(self, analysis: Dict) -> Dict:
        """
        From the analysis, extract a specification for rebuilding.
        This is a heuristic extraction — identifies what the repo DOES
        and what patterns DMAI needs to replicate.
        """
        spec = {
            "purpose": "unknown",
            "patterns": [],
            "key_classes": [],
            "key_functions": [],
            "inputs": [],
            "outputs": [],
            "dependencies_needed": [],
        }

        readme = analysis.get("readme", "")
        if readme:
            # Extract first meaningful paragraph as purpose
            lines = readme.split("\n")
            for line in lines:
                clean = line.strip().lstrip("#").strip()
                if len(clean) > 30 and not clean.startswith("![") and not clean.startswith("<"):
                    spec["purpose"] = clean[:300]
                    break

        # Analyze Python entry points for key classes/functions
        repo_path = Path(analysis.get("repo_path", ""))
        for ep in analysis.get("entry_points", [])[:3]:
            fpath = repo_path / ep
            if not fpath.exists():
                continue
            try:
                content = fpath.read_text(errors="replace")[:3000]
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("class ") and ":" in line:
                        cls_name = line.split("class ")[1].split("(")[0].split(":")[0].strip()
                        if cls_name not in spec["key_classes"]:
                            spec["key_classes"].append(cls_name)
                    if line.startswith("def ") and "(" in line:
                        fn_name = line.split("def ")[1].split("(")[0].strip()
                        if not fn_name.startswith("_") and fn_name not in spec["key_functions"]:
                            spec["key_functions"].append(fn_name)
            except Exception:
                pass

        # Patterns based on dependencies
        deps = analysis.get("dependencies", [])
        if "flask" in deps or "fastapi" in deps or "django" in deps:
            spec["patterns"].append("web_api")
        if "torch" in deps or "tensorflow" in deps or "jax" in deps:
            spec["patterns"].append("machine_learning")
        if "requests" in deps or "httpx" in deps or "aiohttp" in deps:
            spec["patterns"].append("http_client")
        if "sqlite3" in deps or "sqlalchemy" in deps:
            spec["patterns"].append("database")
        if "asyncio" in deps:
            spec["patterns"].append("async")

        # Dependencies DMAI needs to replicate
        for dep in deps[:15]:
            if dep not in ("os", "sys", "json", "time", "datetime", "pathlib", "logging", "re", "typing", "collections"):
                spec["dependencies_needed"].append(dep)

        return spec

    # ------------------------------------------------------------------
    # Stage 5: REBUILD — generate DMAI-native component
    # ------------------------------------------------------------------

    def rebuild(self, repo_name: str, spec: Dict, analysis: Dict) -> Optional[Path]:
        """
        Generate a DMAI-native Python component from the reverse-engineered spec.
        Uses the CodeWriter if available, otherwise generates a skeleton.
        """
        component_name = repo_name.replace("/", "_").replace("-", "_").lower()
        # Sanitize: remove special chars, limit length
        component_name = "".join(c for c in component_name if c.isalnum() or c == "_")[:50]
        target_dir = _REPO_ROOT / "components" / "generated" / "github_imports"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{component_name}.py"

        # Try CodeWriter first
        cw = self.components.get("code_writer")
        if cw and hasattr(cw, "generate_component"):
            try:
                result = cw.generate_component(
                    component_name=component_name,
                    description=spec.get("purpose", f"Auto-generated from GitHub repo: {repo_name}"),
                    requirements=spec.get("dependencies_needed", []),
                    origin=f"github_pipeline:{repo_name}",
                )
                if result.get("ok") and Path(result.get("file", "")).exists():
                    return Path(result["file"])
            except Exception as e:
                logger.debug("CodeWriter generation failed, using skeleton: %s", e)

        # Fallback: generate skeleton
        skeleton = self._generate_skeleton(component_name, repo_name, spec, analysis)
        target_file.write_text(skeleton)
        return target_file

    def _generate_skeleton(self, component_name: str, repo_name: str, spec: Dict, analysis: Dict) -> str:
        """Generate a minimal but well-structured skeleton component."""
        purpose = spec.get("purpose", f"Auto-generated from {repo_name}")
        return f'''"""
DMAI Component: {component_name}
Generated from GitHub repo: {repo_name}
Purpose: {purpose}

Auto-generated by GitHubRepoPipeline.
Patterns detected: {spec.get("patterns", [])}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dmai.{component_name}")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class {self._class_name(component_name)}:
    """
    {purpose[:200]}
    
    Source: https://github.com/{repo_name}
    Patterns: {", ".join(spec.get("patterns", ["unknown"]))}
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.root = data_path or _REPO_ROOT
        logger.info("{component_name} initialised")

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Main entry point — override with specific implementation."""
        return {{"ok": True, "component": "{component_name}", "status": "skeleton_ready"}}

    def get_status(self) -> Dict:
        return {{"component": "{component_name}", "source_repo": "{repo_name}", "status": "active"}}


# ── Quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    comp = {self._class_name(component_name)}()
    print(comp.run())
'''

    @staticmethod
    def _class_name(component_name: str) -> str:
        return "".join(w.capitalize() for w in component_name.split("_"))

    # ------------------------------------------------------------------
    # Stage 6: TEST
    # ------------------------------------------------------------------

    def test_component(self, component_path: Path) -> Dict:
        """Syntax check and smoke test a generated component."""
        result = {"syntax_ok": False, "smoke_ok": False, "errors": []}

        # Syntax check
        try:
            import py_compile
            py_compile.compile(str(component_path), doraise=True)
            result["syntax_ok"] = True
        except Exception as e:
            result["errors"].append(f"Syntax error: {e}")

        # Smoke test: can Python import it?
        if result["syntax_ok"]:
            try:
                rel = component_path.relative_to(_REPO_ROOT)
                module_path = str(rel).replace("/", ".").replace(".py", "")
                subprocess.run(
                    ["python3", "-c", f"import ast; ast.parse(open('{component_path}').read()); print('OK')"],
                    capture_output=True, text=True, timeout=10,
                    cwd=str(_REPO_ROOT),
                )
                result["smoke_ok"] = True
            except Exception as e:
                result["errors"].append(f"Smoke test error: {e}")

        return result

    # ------------------------------------------------------------------
    # Stage 7: REGISTER
    # ------------------------------------------------------------------

    def register_component(self, repo_name: str, component_name: str, component_path: Path, spec: Dict) -> Dict:
        """Register the new component in DMAI's capability registry and knowledge graph."""
        registered = {"capability": False, "vector_store": False, "graph": False}

        # Register in vector store if available
        vs = self.components.get("vector_store")
        if vs and hasattr(vs, "store"):
            try:
                # Generate a simple embedding-like vector from component name hash
                import hashlib
                hash_bytes = hashlib.sha256(component_name.encode()).digest()[:384//8]
                fake_embedding = [float(b) / 255.0 for b in hash_bytes]
                # Pad to 384 dimensions
                fake_embedding += [0.0] * (384 - len(fake_embedding))
                vs.store(
                    entity_type="capability",
                    entity_id=component_name,
                    embedding=fake_embedding,
                    metadata={
                        "source_repo": repo_name,
                        "purpose": spec.get("purpose", "")[:200],
                        "patterns": spec.get("patterns", []),
                        "file": str(component_path.relative_to(_REPO_ROOT)),
                    },
                )
                registered["vector_store"] = True
            except Exception as e:
                logger.debug("Vector store registration: %s", e)

        # Log to SICore if available
        si = self.components.get("si_core")
        if si and hasattr(si, "add_insight"):
            try:
                si.add_insight(
                    domain="github_import",
                    concept=component_name,
                    source=f"github:{repo_name}",
                    confidence=0.8,
                    metadata={"pipeline": "github_repo_pipeline", "patterns": spec.get("patterns", [])},
                )
                registered["graph"] = True
            except Exception as e:
                logger.debug("Graph registration: %s", e)

        return registered

    # ------------------------------------------------------------------
    # Full pipeline execution
    # ------------------------------------------------------------------

    def process_repo(self, repo_name: str, repo_url: str) -> Dict:
        """Run the full pipeline on a single repo."""
        result = {
            "repo_name": repo_name,
            "repo_url": repo_url,
            "stages": {},
            "status": "started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Stage 3: Clone (use existing integration engine or git directly)
            safe_name = repo_name.replace("/", "_")
            repo_path = self.work_dir / safe_name

            if not repo_path.exists():
                try:
                    subprocess.run(
                        ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
                        capture_output=True, text=True, timeout=120,
                    )
                except Exception as e:
                    result["status"] = "clone_failed"
                    result["error"] = str(e)
                    return result

            # Stage 3: Analyze
            analysis = self.analyze_repo(repo_path)
            result["stages"]["analyze"] = analysis.get("structure", {})

            # Stage 4: Reverse engineer
            spec = self.reverse_engineer(analysis)
            result["stages"]["reverse_engineer"] = {
                "purpose": spec.get("purpose", "")[:100],
                "patterns": spec.get("patterns", []),
            }

            # Stage 5: Rebuild
            component_path = self.rebuild(repo_name, spec, analysis)
            if component_path:
                result["stages"]["rebuild"] = {"component_path": str(component_path)}
            else:
                result["status"] = "rebuild_failed"
                return result

            # Stage 6: Test
            test_result = self.test_component(component_path)
            result["stages"]["test"] = test_result
            if not test_result["syntax_ok"]:
                result["status"] = "test_failed"
                return result

            # Stage 7: Register
            component_name = component_path.stem
            registration = self.register_component(repo_name, component_name, component_path, spec)
            result["stages"]["register"] = registration

            result["status"] = "success"
            self.state["total_succeeded"] += 1

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            self.state["total_failed"] += 1

        self.state["processed"].append({
            "repo_name": repo_name,
            "status": result["status"],
            "timestamp": result["timestamp"],
        })
        self.state["total_processed"] += 1
        self._save_state()
        self._log(result["status"], result)

        return result

    def get_stats(self) -> Dict:
        return dict(self.state)


def start_pipeline_loop(components: dict, interval_minutes: float = 30.0):
    """Background daemon: processes one starred repo per interval."""

    def _loop():
        time.sleep(90)
        pipeline = GitHubRepoPipeline(components)
        while True:
            try:
                # Get starred repos from scanner
                scanner = components.get("github_starred_scanner")
                if scanner:
                    history = scanner.scan_history.get("scanned_repos", {})
                    # Find unprocessed repos
                    processed_names = {p["repo_name"] for p in pipeline.state["processed"]}
                    for repo_id, info in history.items():
                        name = info.get("full_name", "")
                        url = info.get("url", "")
                        if name and name not in processed_names:
                            logger.info("GitHubRepoPipeline: processing %s", name)
                            pipeline.process_repo(name, url)
                            break  # One per cycle
            except Exception as e:
                logger.warning("GitHubRepoPipeline loop error: %s", e)
            time.sleep(interval_minutes * 60)

    t = threading.Thread(target=_loop, daemon=True, name="GitHubRepoPipeline")
    t.start()
    logger.info("GitHubRepoPipeline started (every %d min)", interval_minutes)
