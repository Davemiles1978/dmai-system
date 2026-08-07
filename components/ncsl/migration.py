"""
NCSL Self-Migration Module
Monitors NCSL functionality %. When >=95%, auto-migrates Python codebase.
"""
import ast, os, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple


class NCSLMigrationEngine:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.state_file = self.project_root / "data" / "ncsl_migration_state.json"
        self.state = self._load_state()
    
    def _load_state(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {
            "started_at": None, "completed_at": None,
            "modules_total": 0, "modules_migrated": 0,
            "modules_failed": 0, "modules_skipped": 0,
            "migrated_modules": [], "failed_modules": {},
            "status": "waiting"
        }
    
    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2, default=str))
    
    def get_functionality_pct(self) -> float:
        try:
            from components.ncsl.token_table import VERSION
            features = {
                "compiler": 15, "vm": 15, "decompiler": 10,
                "strings": 5, "integers": 5, "floats": 5,
                "lists_dicts": 5, "control_flow": 8, "functions": 5,
                "calls": 5, "call_stack": 12, "imports": 12,
                "exceptions": 10, "classes": 10, "expression_tree": 10,
                "type_system": 8, "async": 8, "optimizer": 8,
                "jit": 8, "sixg": 5, "predictive": 3,
            }
            done_map = {
                "compiler": True, "vm": True, "decompiler": True,
                "strings": True, "integers": True, "floats": True,
                "lists_dicts": True, "control_flow": True, "functions": True,
                "calls": True, "call_stack": False, "imports": False,
                "exceptions": False, "classes": False, "expression_tree": False,
                "type_system": False, "async": False, "optimizer": False,
                "jit": False, "sixg": True, "predictive": True,
            }
            done = sum(w for f, w in features.items() if done_map.get(f, False))
            total = sum(features.values())
            return round(done / total * 100, 1)
        except:
            return 50.0
    
    def check_readiness(self) -> Tuple[bool, float]:
        pct = self.get_functionality_pct()
        return pct >= 95.0, pct
    
    def scan_modules(self) -> List[Path]:
        python_files = []
        exclude = {'__pycache__', '.git', 'node_modules', 'data', 'venv', 'ncsl'}
        for py_file in self.project_root.rglob("*.py"):
            if any(d in py_file.parts for d in exclude):
                continue
            python_files.append(py_file)
        python_files.sort(key=lambda f: f.stat().st_size)
        return python_files
    
    def migrate_module(self, py_path: Path) -> Dict:
        result = {
            "module": str(py_path.relative_to(self.project_root)),
            "python_size": py_path.stat().st_size,
            "status": "pending", "error": None
        }
        try:
            from components.ncsl import NCSLEngine
            engine = NCSLEngine()
            py_source = py_path.read_text()
            ncsl_binary = engine.compile(py_source)
            decompiled = engine.decompile(ncsl_binary)
            
            if not decompiled or len(decompiled.strip()) < 5:
                result["status"] = "failed"
                result["error"] = "Decompiled output too short"
                return result
            
            py_tree = ast.parse(py_source)
            ncsl_tree = ast.parse(decompiled)
            py_funcs = len([n for n in ast.walk(py_tree) if isinstance(n, ast.FunctionDef)])
            ncsl_funcs = len([n for n in ast.walk(ncsl_tree) if isinstance(n, ast.FunctionDef)])
            
            if py_funcs != ncsl_funcs:
                result["status"] = "failed"
                result["error"] = f"Function count mismatch: {py_funcs} vs {ncsl_funcs}"
                return result
            
            # Backup original
            backup = py_path.with_suffix(".py.ncsl_backup")
            backup.write_text(py_source)
            
            # Write NCSL binary
            ncsl_path = py_path.with_suffix(".ncsl")
            ncsl_path.write_bytes(ncsl_binary)
            
            # Write Python wrapper
            wrapper = f'''"""
Auto-migrated: {py_path.name} → NCSL
"""
from components.ncsl import NCSLEngine
import pathlib
_engine = NCSLEngine()
_data = pathlib.Path(__file__).parent / "{ncsl_path.name}"
_result = _engine.execute(_data.read_bytes())
'''
            py_path.write_text(wrapper)
            
            result["status"] = "migrated"
            result["ncsl_size"] = len(ncsl_binary)
            result["compression_pct"] = round((1 - len(ncsl_binary) / len(py_source.encode())) * 100, 1)
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
        return result
    
    def run_migration(self) -> Dict:
        ready, pct = self.check_readiness()
        if not ready:
            return {"status": "not_ready", "functionality_pct": pct}
        
        modules = self.scan_modules()
        self.state["modules_total"] = len(modules)
        self.state["status"] = "migrating"
        self.state["started_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state()
        
        results = []
        for mod in modules:
            self.state["current_module"] = str(mod.relative_to(self.project_root))
            self._save_state()
            r = self.migrate_module(mod)
            results.append(r)
            if r["status"] == "migrated":
                self.state["modules_migrated"] += 1
            elif r["status"] == "failed":
                self.state["modules_failed"] += 1
                self.state["failed_modules"][r["module"]] = r["error"]
        
        self.state["status"] = "complete"
        self.state["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state()
        return {"status": "complete", "total": len(modules),
                "migrated": self.state["modules_migrated"],
                "failed": self.state["modules_failed"], "results": results}


def start_migration_monitor(project_root: str, check_interval_minutes: int = 60):
    import threading, time
    engine = NCSLMigrationEngine(project_root)
    def loop():
        while True:
            try:
                ready, pct = engine.check_readiness()
                if ready and engine.state["status"] == "waiting":
                    print(f"NCSL at {pct}% — starting auto-migration")
                    engine.run_migration()
                elif engine.state["status"] == "complete":
                    break
            except Exception as e:
                print(f"Migration monitor: {e}")
            time.sleep(check_interval_minutes * 60)
    t = threading.Thread(target=loop, daemon=True, name="ncsl-migration")
    t.start()
    return t
