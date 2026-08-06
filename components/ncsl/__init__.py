"""
NCSL — Neural Coding Symbolic Language v3.0
DMAI's native compressed language for storage, transmission, and execution.

Pipeline: Python source → NCSL binary → VM execution or decompile back to Python
6G-ready: THz photonic mapping, predictive zero-byte mode, ISAC frames
"""
from .compiler import NCSLCompiler
from .vm import NCSLVM
from .decompiler import NCSLDecompiler


class NCSLEngine:
    """Complete NCSL pipeline."""
    
    def __init__(self):
        self.compiler = NCSLCompiler()
        self.vm = NCSLVM()
        self.decompiler = NCSLDecompiler()
    
    def compile(self, python_source: str) -> bytes:
        return self.compiler.compile(python_source)
    
    def execute(self, ncsl_binary: bytes):
        self.vm.load(ncsl_binary)
        return self.vm.run()
    
    def decompile(self, ncsl_binary: bytes) -> str:
        return self.decompiler.decompile(ncsl_binary)
    
    def stats(self, python_source: str) -> dict:
        py_bytes = len(python_source.encode('utf-8'))
        ncsl_bytes = len(self.compile(python_source))
        return {
            "python_bytes": py_bytes,
            "ncsl_bytes": ncsl_bytes,
            "compression_pct": round((1 - ncsl_bytes / py_bytes) * 100, 1),
        }
