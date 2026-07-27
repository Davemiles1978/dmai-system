import os, sys
from typing import Callable, Dict, Any, List

class CodeBlockFactory:
    """Generates, tests, and registers new Python functions as permanent tools."""
    def __init__(self, sandbox_dir: str = "/mnt/code_tools"):
        self.registry: Dict[str, Callable] = {}
        self.sandbox_dir = sandbox_dir
        os.makedirs(sandbox_dir, exist_ok=True)
        if sandbox_dir not in sys.path:
            sys.path.append(sandbox_dir)

    def create_tool(self, spec: Dict[str, Any]) -> Callable:
        name = spec["name"]
        code = spec["code_template"]
        compiled = compile(code, f"<{name}>", "exec")
        exec_globals = {}
        exec(compiled, exec_globals)
        func = exec_globals.get(name)
        if not func:
            raise ValueError(f"Function '{name}' not found in provided code.")
        for i, test in enumerate(spec.get("test_cases", []), 1):
            args = test.get("args", [])
            kwargs = test.get("kwargs", {})
            expected = test["expected"]
            result = func(*args, **kwargs)
            assert result == expected, f"Test {i} failed: expected {expected}, got {result}"
        module_path = os.path.join(self.sandbox_dir, f"{name}.py")
        with open(module_path, "w") as f:
            f.write(code)
        self.registry[name] = func
        return func

    def list_tools(self) -> List[str]:
        return list(self.registry.keys())

    def remove_tool(self, name: str):
        if name in self.registry:
            del self.registry[name]
            module_path = os.path.join(self.sandbox_dir, f"{name}.py")
            if os.path.exists(module_path):
                os.remove(module_path)
