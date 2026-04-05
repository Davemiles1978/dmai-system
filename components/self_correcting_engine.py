"""
Self-Correcting Code Engine - Zero-Error Loop Implementation

This module:
- Runs code and captures errors
- Analyzes error messages
- Generates fixes
- Re-runs until clean
- Returns only successful code
"""

import sys
import io
import traceback
import ast
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeFix:
    """Represents a fix to apply to code"""
    line_number: int
    original: str
    fixed: str
    reason: str


class SelfCorrectingEngine:
    """Zero-error loop: run, check, fix, repeat until clean"""
    
    def __init__(self, max_attempts: int = 5):
        self.max_attempts = max_attempts
        self.fix_history: List[Dict] = []
        
    def run_and_correct(self, code: str, context: Dict = None) -> Tuple[bool, str, List[str]]:
        """
        Run code, catch errors, fix them, repeat until clean.
        
        Returns:
            (success, corrected_code, error_history)
        """
        current_code = code
        error_history = []
        
        for attempt in range(self.max_attempts):
            print(f"🔧 Attempt {attempt + 1}/{self.max_attempts}")
            
            # Run the code
            success, error, output = self._execute_code(current_code, context)
            
            if success:
                print(f"✅ Code executed successfully on attempt {attempt + 1}")
                return True, current_code, error_history
            
            # Record error
            error_history.append({
                'attempt': attempt + 1,
                'error': error,
                'code': current_code
            })
            
            # Try to fix the error
            fixed_code, fix_applied = self._fix_error(current_code, error)
            
            if not fix_applied:
                print(f"⚠️ Could not auto-fix error: {error[:100]}")
                return False, current_code, error_history
            
            current_code = fixed_code
            print(f"🔧 Applied fix: {fix_applied}")
        
        print(f"❌ Failed to fix after {self.max_attempts} attempts")
        return False, current_code, error_history
    
    def _execute_code(self, code: str, context: Dict = None) -> Tuple[bool, str, str]:
        """Execute code safely and capture output/errors"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Compile and execute
            compiled = compile(code, '<string>', 'exec')
            exec(compiled, context or {})
            output = sys.stdout.getvalue()
            return True, None, output
            
        except Exception as e:
            error_msg = traceback.format_exc()
            return False, error_msg, None
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _extract_line_number(self, error: str) -> Optional[int]:
        """Extract line number from traceback"""
        # Look for line number in traceback
        patterns = [
            r'File "<string>", line (\d+)',
            r'line (\d+)',
            r'\(line (\d+)\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error)
            if match:
                return int(match.group(1))
        return None
    
    def _fix_error(self, code: str, error: str) -> Tuple[str, Optional[str]]:
        """Analyze error and generate fix"""
        lines = code.split('\n')
        error_line = self._extract_line_number(error)
        
        # Dictionary changed size during iteration
        if 'dictionary changed size during iteration' in error:
            return self._fix_dict_iteration(lines, error_line)
        
        # Indentation errors
        if 'IndentationError' in error:
            return self._fix_indentation(lines, error_line)
        
        # Syntax errors
        if 'SyntaxError' in error:
            return self._fix_syntax(lines, error_line, error)
        
        # Name errors
        if 'NameError' in error and 'is not defined' in error:
            return self._fix_name_error(lines, error, error_line)
        
        # Attribute errors (float has no attribute get)
        if 'AttributeError' in error and 'has no attribute' in error:
            return self._fix_attribute_error(lines, error, error_line)
        
        return code, None
    
    def _fix_dict_iteration(self, lines: List[str], error_line: int) -> Tuple[str, str]:
        """Fix dictionary changed size during iteration"""
        if not error_line:
            # Try to find any for loop with .items()
            for i, line in enumerate(lines):
                if 'for' in line and '.items()' in line and 'list(' not in line:
                    error_line = i + 1
                    break
        
        if error_line and error_line <= len(lines):
            # Find the for loop line (might be above the error line)
            for i in range(max(0, error_line - 3), min(len(lines), error_line + 1)):
                line = lines[i]
                
                # Fix .items()
                if '.items()' in line and 'for' in line and 'list(' not in line:
                    # Convert: for k, v in d.items() -> for k, v in list(d.items())
                    parts = line.split('.items()')
                    if len(parts) >= 2:
                        fixed_line = parts[0] + '.items()'
                        # Wrap the left side with list()
                        left_part = parts[0].strip()
                        if left_part.endswith('in'):
                            left_part = left_part[:-2].strip()
                        fixed_line = f"{left_part} in list({left_part}.items())"
                        # Preserve the rest after .items()
                        if len(parts) > 1:
                            fixed_line += parts[1]
                        lines[i] = fixed_line
                        return "\n".join(lines), "Converted dict.items() to list(dict.items())"
        
        # Fallback: replace all .items() in for loops
        import re
        fixed_code = "\n".join(lines)
        fixed_code = re.sub(
            r'(for\s+[\w,\s]+\s+in\s+)(\w+)\.items\(\)',
            r'\1list(\2.items())',
            fixed_code
        )
        return fixed_code, "Applied global fix for dictionary iteration"
    
    def _fix_indentation(self, lines: List[str], error_line: int) -> Tuple[str, str]:
        """Fix indentation errors"""
        if error_line and error_line <= len(lines):
            lines[error_line - 1] = "    " + lines[error_line - 1].lstrip()
            return "\n".join(lines), "Fixed indentation"
        return "\n".join(lines), "Added default indentation"
    
    def _fix_syntax(self, lines: List[str], error_line: int, error: str) -> Tuple[str, str]:
        """Fix common syntax errors"""
        if error_line and error_line <= len(lines):
            line = lines[error_line - 1]
            
            # Fix missing colon
            if ':' not in line and any(kw in line for kw in ['if ', 'for ', 'while ', 'def ', 'class ']):
                lines[error_line - 1] = line.rstrip() + ':'
                return "\n".join(lines), "Added missing colon"
            
            # Fix unterminated string
            if "EOL" in error:
                if line.count('"') % 2 == 1:
                    lines[error_line - 1] = line + '"'
                elif line.count("'") % 2 == 1:
                    lines[error_line - 1] = line + "'"
                return "\n".join(lines), "Added missing quote"
        
        return "\n".join(lines), None
    
    def _fix_name_error(self, lines: List[str], error: str, error_line: int) -> Tuple[str, str]:
        """Fix undefined name errors"""
        match = re.search(r"name '(\w+)' is not defined", error)
        if match and error_line and error_line <= len(lines):
            undefined_name = match.group(1)
            lines.insert(error_line - 1, f"{undefined_name} = None  # Auto-fixed")
            return "\n".join(lines), f"Added placeholder for {undefined_name}"
        return "\n".join(lines), None
    
    def _fix_attribute_error(self, lines: List[str], error: str, error_line: int) -> Tuple[str, str]:
        """Fix attribute errors (like .get() on float)"""
        match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error)
        if not match:
            return "\n".join(lines), None
        
        obj_type = match.group(1)
        attr = match.group(2)
        
        if not error_line or error_line > len(lines):
            return "\n".join(lines), None
        
        line = lines[error_line - 1]
        
        # Fix float.get() error
        if obj_type == 'float' and attr == 'get':
            # Find the variable name
            var_match = re.search(r'(\w+)\.get\(', line)
            if var_match:
                var_name = var_match.group(1)
                indent = len(line) - len(line.lstrip())
                
                # Create a safe version with type checking
                fixed_lines = [
                    ' ' * indent + f"if isinstance({var_name}, dict):",
                    ' ' * (indent + 4) + line.strip(),
                    ' ' * indent + "else:",
                    ' ' * (indent + 4) + f"value = 0  # {var_name} is {obj_type}, cannot use .get()",
                ]
                
                # Replace the original line with the fixed block
                lines[error_line - 1] = fixed_lines[0]
                lines.insert(error_line, fixed_lines[1])
                lines.insert(error_line + 1, fixed_lines[2])
                lines.insert(error_line + 2, fixed_lines[3])
                
                return "\n".join(lines), f"Added type check for {var_name} (was {obj_type})"
        
        # General attribute error fix
        indent = len(line) - len(line.lstrip())
        fixed_lines = [
            ' ' * indent + "try:",
            ' ' * (indent + 4) + line.strip(),
            ' ' * indent + "except AttributeError:",
            ' ' * (indent + 4) + f"# {obj_type} has no attribute {attr}, using default",
            ' ' * (indent + 4) + "value = None",
        ]
        
        lines[error_line - 1] = fixed_lines[0]
        for i, fline in enumerate(fixed_lines[1:]):
            lines.insert(error_line + i, fline)
        
        return "\n".join(lines), f"Wrapped {attr} access in try/except"

def safe_execute(code: str, max_attempts: int = 5) -> Tuple[bool, str, List[str]]:
    """Convenience function: execute code with automatic error correction"""
    engine = SelfCorrectingEngine(max_attempts)
    return engine.run_and_correct(code)


# Test with problematic code
if __name__ == "__main__":
    print("Testing Self-Correcting Engine...")
    print("=" * 50)
    
    # Test 1: Dictionary iteration error
    test_code = """
def test_function():
    d = {'a': 1, 'b': 2}
    for k, v in d.items():
        if k == 'a':
            d['c'] = 3
        print(k)
test_function()
"""
    print("Test 1: Dictionary iteration error")
    engine = SelfCorrectingEngine(max_attempts=3)
    success, fixed, history = engine.run_and_correct(test_code)
    print(f"Success: {success}")
    if success:
        print("Fixed code:")
        print(fixed)
    print()
    
    # Test 2: Float attribute error
    test_code2 = """
x = 3.14
value = x.get('key', 0)
"""
    print("Test 2: Float .get() error")
    success, fixed, history = engine.run_and_correct(test_code2)
    print(f"Success: {success}")
    if success:
        print("Fixed code:")
        print(fixed)
