# components/capability_integrator.py
"""
DMAI Capability Integrator - Extracts and integrates actual capabilities from repositories
Supports: Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Shell, JSON, YAML, TOML, XML, Markdown, Text
"""

import os
import ast
import json
import re
import shutil
import tempfile
import subprocess
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Optional imports with fallbacks
try:
    import yaml
except ImportError:
    yaml = None

try:
    import xml.etree.ElementTree as ET
except ImportError:
    ET = None

logger = logging.getLogger(__name__)


class CapabilityIntegrator:
    """
    Extracts actual functions/classes from repositories and integrates them into DMAI.
    Supports multiple languages: Python, TypeScript, JavaScript, Go, Rust, Java, C/C++, Shell,
    and configuration files: JSON, YAML, TOML, XML, Markdown, Text.
    """
    
    def __init__(self, dmai_app):
        self.dmai = dmai_app
        self.capabilities_dir = Path("components/capabilities")
        self.capabilities_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = Path("data/capabilities/registry.json")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.registry = self._load_registry()
        
        # Runtime mode tracking
        self.autonomous_capabilities = []
        self.ondemand_capabilities = []
        
    def _load_registry(self) -> Dict:
        """Load existing capability registry"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'capabilities': {},
            'sources': {},
            'last_updated': None,
            'total_capabilities': 0
        }
    
    def _save_registry(self):
        """Save capability registry"""
        self.registry['last_updated'] = datetime.now().isoformat()
        self.registry['total_capabilities'] = len(self.registry['capabilities'])
        
        # Save to JSON
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
        
        # ============================================================
        # PERSISTENCE GUARANTEE: Also save to SQLite
        # ============================================================
        if hasattr(self.dmai, 'si_core') and hasattr(self.dmai.si_core, 'sqlite') and self.dmai.si_core.sqlite:
            try:
                for cap_id, cap in self.registry['capabilities'].items():
                    self.dmai.si_core.sqlite.save_capability(cap_id, cap)
                logger.info(f"💾 Saved {len(self.registry['capabilities'])} capabilities to SQLite")
            except Exception as e:
                logger.error(f"SQLite capability save failed: {e}")
    
    def process_repository(self, repo_url: str) -> Dict:
        """
        Main entry point - process a GitHub repository and integrate its capabilities
        
        Returns:
            Dict with integration results including capabilities added, neurons created, etc.
        """
        result = {
            'success': True,
            'repo_url': repo_url,
            'repo_name': repo_url.split('/')[-1].replace('.git', ''),
            'capabilities_found': [],
            'capabilities_integrated': [],
            'capabilities_skipped': [],
            'neurons_created': [],
            'files_copied': [],
            'errors': []
        }
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Step 1: Clone repository
            logger.info(f"📥 Cloning {repo_url}...")
            clone_result = subprocess.run(
                ['git', 'clone', '--depth', '1', repo_url, temp_dir],
                capture_output=True, text=True, timeout=120
            )
            
            if clone_result.returncode != 0:
                result['success'] = False
                result['errors'].append(f"Clone failed: {clone_result.stderr}")
                return result
            
            # Step 2: Extract all capabilities from ALL supported file types
            extracted = self._extract_capabilities_from_repo(temp_dir, repo_url)
            result['capabilities_found'] = extracted
            
            logger.info(f"🔍 Found {len(extracted)} capabilities in {repo_url}")
            
            # Step 3: Compare against existing and integrate new ones
            for capability in extracted:
                integration_result = self._integrate_capability(
                    capability, 
                    temp_dir, 
                    result['repo_name']
                )
                
                if integration_result['integrated']:
                    result['capabilities_integrated'].append(integration_result)
                    
                    # Create neuron for this capability
                    neuron_id = self._create_capability_neuron(integration_result, repo_url)
                    if neuron_id:
                        result['neurons_created'].append(neuron_id)
                        
                    if integration_result.get('file_copied'):
                        result['files_copied'].append(integration_result['file_copied'])
                else:
                    result['capabilities_skipped'].append(integration_result)
            
            # Step 4: Save updated registry
            self._save_registry()
            
            # Step 5: Record source in registry
            self.registry['sources'][repo_url] = {
                'processed_at': datetime.now().isoformat(),
                'capabilities_integrated': len(result['capabilities_integrated']),
                'repo_name': result['repo_name']
            }
            self._save_registry()
            
        except Exception as e:
            logger.error(f"Error processing repository: {e}")
            result['success'] = False
            result['errors'].append(str(e))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    def _extract_capabilities_from_repo(self, repo_path: str, source_url: str) -> List[Dict]:
        """Extract capabilities from all supported file types in the repository"""
        capabilities = []
        
        # Define file patterns and their parsers
        parsers = {
            # Code files
            '.py': self._parse_python_file,
            '.ts': self._parse_typescript_file,
            '.tsx': self._parse_typescript_file,
            '.js': self._parse_javascript_file,
            '.jsx': self._parse_javascript_file,
            '.go': self._parse_go_file,
            '.rs': self._parse_rust_file,
            '.java': self._parse_java_file,
            '.cpp': self._parse_cpp_file,
            '.cc': self._parse_cpp_file,
            '.cxx': self._parse_cpp_file,
            '.c': self._parse_c_file,
            '.h': self._parse_header_file,
            '.hpp': self._parse_header_file,
            '.sh': self._parse_shell_file,
            '.bash': self._parse_shell_file,
            
            # Configuration files
            '.json': self._parse_json_file,
            '.yaml': self._parse_yaml_file,
            '.yml': self._parse_yaml_file,
            '.toml': self._parse_toml_file,
            '.xml': self._parse_xml_file,
            
            # Documentation files
            '.md': self._parse_markdown_file,
            '.markdown': self._parse_markdown_file,
            '.txt': self._parse_text_file,
            '.rst': self._parse_rst_file,
        }
        
        # Walk through all files
        for file_path in Path(repo_path).rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip test files, virtual environments, and cache
            path_str = str(file_path).lower()
            if any(skip in path_str for skip in ['test', 'spec', '__pycache__', 'node_modules', 
                                                  'venv', 'env', '.git', 'dist', 'build']):
                continue
            
            suffix = file_path.suffix.lower()
            if suffix in parsers:
                try:
                    extracted = parsers[suffix](file_path, source_url)
                    if extracted:
                        capabilities.extend(extracted)
                except Exception as e:
                    logger.warning(f"Could not parse {file_path}: {e}")
        
        return capabilities

    # ============================================================
    # PYTHON PARSER (AST-based)
    # ============================================================
    
    def _parse_python_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse Python file using AST"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            module_doc = ast.get_docstring(tree) or ""
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                    cap = self._extract_python_class(node, file_path, content, module_doc, source_url)
                    if cap:
                        capabilities.append(cap)
                elif isinstance(node, ast.FunctionDef) and self._is_top_level_function(node, tree):
                    if not node.name.startswith('_'):
                        cap = self._extract_python_function(node, file_path, content, module_doc, source_url)
                        if cap:
                            capabilities.append(cap)
        except Exception as e:
            logger.debug(f"Python parse error {file_path}: {e}")
        
        return capabilities
    
    def _extract_python_class(self, node: ast.ClassDef, filepath: Path, 
                               content: str, module_doc: str, source_url: str) -> Optional[Dict]:
        """Extract a Python class as a capability"""
        class_name = node.name
        
        if class_name.startswith('_'):
            return None
        
        docstring = ast.get_docstring(node) or f"Class {class_name} from {filepath.name}"
        
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if not item.name.startswith('_'):
                    methods.append({
                        'name': item.name,
                        'docstring': ast.get_docstring(item) or "",
                        'args': [arg.arg for arg in item.args.args],
                        'is_async': isinstance(item, ast.AsyncFunctionDef)
                    })
        
        class_source = self._get_node_source(content, node)
        capability_type = self._infer_capability_type(class_name, methods, module_doc)
        capability_id = hashlib.md5(f"{class_name}_{filepath.stem}".encode()).hexdigest()[:12]
        
        return {
            'id': capability_id,
            'name': class_name,
            'type': 'class',
            'capability_type': capability_type,
            'description': docstring[:200],
            'methods': methods,
            'source_file': str(filepath),
            'source_code': class_source,
            'source_url': source_url,
            'dependencies': self._extract_dependencies(filepath),
            'imports': self._extract_imports(content),
            'language': 'python'
        }
    
    def _extract_python_function(self, node: ast.FunctionDef, filepath: Path,
                                  content: str, module_doc: str, source_url: str) -> Optional[Dict]:
        """Extract a Python function as a capability"""
        func_name = node.name
        
        if func_name.startswith('_'):
            return None
        
        docstring = ast.get_docstring(node) or f"Function {func_name} from {filepath.name}"
        func_source = self._get_node_source(content, node)
        capability_type = self._infer_capability_type(func_name, [], module_doc)
        capability_id = hashlib.md5(f"{func_name}_{filepath.stem}".encode()).hexdigest()[:12]
        
        return {
            'id': capability_id,
            'name': func_name,
            'type': 'function',
            'capability_type': capability_type,
            'description': docstring[:200],
            'source_file': str(filepath),
            'source_code': func_source,
            'source_url': source_url,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'args': [arg.arg for arg in node.args.args],
            'dependencies': self._extract_dependencies(filepath),
            'imports': self._extract_imports(content),
            'language': 'python'
        }
    
    def _is_top_level_function(self, node: ast.FunctionDef, tree: ast.Module) -> bool:
        """Check if function is defined at module level"""
        for item in tree.body:
            if isinstance(item, ast.FunctionDef) and item == node:
                return True
        return False
    
    def _get_node_source(self, content: str, node: ast.AST) -> str:
        """Extract the source code for an AST node"""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
        return '\n'.join(lines[start_line:end_line])

    # ============================================================
    # JAVASCRIPT / TYPESCRIPT PARSERS
    # ============================================================
    
    def _parse_typescript_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse TypeScript file using regex patterns"""
        return self._parse_js_ts_common(file_path, source_url, 'typescript')
    
    def _parse_javascript_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse JavaScript file using regex patterns"""
        return self._parse_js_ts_common(file_path, source_url, 'javascript')
    
    def _parse_js_ts_common(self, file_path: Path, source_url: str, lang: str) -> List[Dict]:
        """Common parser for JavaScript and TypeScript"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract classes
            class_pattern = r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*\{'
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                class_name = match.group(1)
                if not class_name.startswith('_'):
                    methods = []
                    method_pattern = r'(?:public|private|protected|async)?\s*(\w+)\s*\([^)]*\)\s*[:{]\s*(?:[^{}]*|\{[^{}]*\})*?\}'
                    
                    capabilities.append({
                        'id': hashlib.md5(f"{class_name}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': class_name,
                        'type': 'class',
                        'capability_type': self._infer_capability_type(class_name, methods, ""),
                        'description': f"{lang} class: {class_name}",
                        'source_file': str(file_path),
                        'source_code': match.group(0)[:500],
                        'source_url': source_url,
                        'methods': methods,
                        'language': lang
                    })
            
            # Extract exported functions
            func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                if not func_name.startswith('_'):
                    capabilities.append({
                        'id': hashlib.md5(f"{func_name}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': func_name,
                        'type': 'function',
                        'capability_type': self._infer_capability_type(func_name, [], ""),
                        'description': f"{lang} function: {func_name}",
                        'source_file': str(file_path),
                        'source_code': match.group(0),
                        'source_url': source_url,
                        'language': lang
                    })
            
            # Extract const arrow functions
            arrow_pattern = r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>'
            for match in re.finditer(arrow_pattern, content):
                func_name = match.group(1)
                if not func_name.startswith('_'):
                    capabilities.append({
                        'id': hashlib.md5(f"{func_name}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': func_name,
                        'type': 'function',
                        'capability_type': self._infer_capability_type(func_name, [], ""),
                        'description': f"{lang} arrow function: {func_name}",
                        'source_file': str(file_path),
                        'source_code': match.group(0),
                        'source_url': source_url,
                        'language': lang
                    })
            
            # Extract interfaces (TypeScript)
            interface_pattern = r'(?:export\s+)?interface\s+(\w+)'
            for match in re.finditer(interface_pattern, content):
                interface_name = match.group(1)
                capabilities.append({
                    'id': hashlib.md5(f"interface_{interface_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': interface_name,
                    'type': 'interface',
                    'capability_type': 'data_structure',
                    'description': f"{lang} interface: {interface_name}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': lang
                })
                    
        except Exception as e:
            logger.debug(f"JS/TS parse error {file_path}: {e}")
        
        return capabilities

    # ============================================================
    # GO PARSER
    # ============================================================
    
    def _parse_go_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse Go file"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract structs
            struct_pattern = r'type\s+(\w+)\s+struct\s*\{([^}]*)\}'
            for match in re.finditer(struct_pattern, content, re.DOTALL):
                struct_name = match.group(1)
                if struct_name and struct_name[0].isupper():
                    capabilities.append({
                        'id': hashlib.md5(f"{struct_name}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': struct_name,
                        'type': 'struct',
                        'capability_type': self._infer_capability_type(struct_name, [], ""),
                        'description': f"Go struct: {struct_name}",
                        'source_file': str(file_path),
                        'source_code': match.group(0)[:500],
                        'source_url': source_url,
                        'language': 'go'
                    })
            
            # Extract functions
            func_pattern = r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                if func_name and func_name[0].isupper():
                    capabilities.append({
                        'id': hashlib.md5(f"{func_name}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': func_name,
                        'type': 'function',
                        'capability_type': self._infer_capability_type(func_name, [], ""),
                        'description': f"Go function: {func_name}",
                        'source_file': str(file_path),
                        'source_code': match.group(0),
                        'source_url': source_url,
                        'language': 'go'
                    })
            
            # Extract interfaces
            interface_pattern = r'type\s+(\w+)\s+interface\s*\{'
            for match in re.finditer(interface_pattern, content):
                interface_name = match.group(1)
                if interface_name[0].isupper():
                    capabilities.append({
                        'id': hashlib.md5(f"interface_{interface_name}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': interface_name,
                        'type': 'interface',
                        'capability_type': 'data_structure',
                        'description': f"Go interface: {interface_name}",
                        'source_file': str(file_path),
                        'source_url': source_url,
                        'language': 'go'
                    })
        except Exception as e:
            logger.debug(f"Go parse error {file_path}: {e}")
        
        return capabilities

    # ============================================================
    # RUST PARSER
    # ============================================================
    
    def _parse_rust_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse Rust file"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract structs
            struct_pattern = r'(?:pub\s+)?struct\s+(\w+)\s*\{'
            for match in re.finditer(struct_pattern, content):
                struct_name = match.group(1)
                capabilities.append({
                    'id': hashlib.md5(f"{struct_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': struct_name,
                    'type': 'struct',
                    'capability_type': self._infer_capability_type(struct_name, [], ""),
                    'description': f"Rust struct: {struct_name}",
                    'source_file': str(file_path),
                    'source_code': match.group(0),
                    'source_url': source_url,
                    'language': 'rust'
                })
            
            # Extract impl blocks
            impl_pattern = r'impl\s+(?:(\w+)\s+for\s+)?(\w+)\s*\{'
            for match in re.finditer(impl_pattern, content):
                trait_name = match.group(1)
                type_name = match.group(2)
                impl_name = f"{trait_name}_for_{type_name}" if trait_name else type_name
                capabilities.append({
                    'id': hashlib.md5(f"impl_{impl_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': f"{impl_name}Impl",
                    'type': 'impl',
                    'capability_type': self._infer_capability_type(type_name, [], ""),
                    'description': f"Rust impl for: {impl_name}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'rust'
                })
            
            # Extract pub functions
            func_pattern = r'pub\s+(?:async\s+)?fn\s+(\w+)\s*\([^)]*\)'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                capabilities.append({
                    'id': hashlib.md5(f"{func_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': func_name,
                    'type': 'function',
                    'capability_type': self._infer_capability_type(func_name, [], ""),
                    'description': f"Rust function: {func_name}",
                    'source_file': str(file_path),
                    'source_code': match.group(0),
                    'source_url': source_url,
                    'language': 'rust'
                })
            
            # Extract traits
            trait_pattern = r'pub\s+trait\s+(\w+)'
            for match in re.finditer(trait_pattern, content):
                trait_name = match.group(1)
                capabilities.append({
                    'id': hashlib.md5(f"trait_{trait_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': trait_name,
                    'type': 'trait',
                    'capability_type': 'interface',
                    'description': f"Rust trait: {trait_name}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'rust'
                })
        except Exception as e:
            logger.debug(f"Rust parse error {file_path}: {e}")
        
        return capabilities

    # ============================================================
    # JAVA PARSER
    # ============================================================
    
    def _parse_java_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse Java file"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract public classes
            class_pattern = r'public\s+(?:abstract\s+)?(?:final\s+)?class\s+(\w+)'
            for match in re.finditer(class_pattern, content):
                class_name = match.group(1)
                capabilities.append({
                    'id': hashlib.md5(f"{class_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': class_name,
                    'type': 'class',
                    'capability_type': self._infer_capability_type(class_name, [], ""),
                    'description': f"Java class: {class_name}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'java'
                })
            
            # Extract interfaces
            interface_pattern = r'public\s+interface\s+(\w+)'
            for match in re.finditer(interface_pattern, content):
                interface_name = match.group(1)
                capabilities.append({
                    'id': hashlib.md5(f"interface_{interface_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': interface_name,
                    'type': 'interface',
                    'capability_type': 'data_structure',
                    'description': f"Java interface: {interface_name}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'java'
                })
        except Exception as e:
            logger.debug(f"Java parse error {file_path}: {e}")
        
        return capabilities

    # ============================================================
    # C / C++ PARSERS
    # ============================================================
    
    def _parse_cpp_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse C++ file"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract classes
            class_pattern = r'class\s+(\w+)'
            for match in re.finditer(class_pattern, content):
                class_name = match.group(1)
                capabilities.append({
                    'id': hashlib.md5(f"{class_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': class_name,
                    'type': 'class',
                    'capability_type': self._infer_capability_type(class_name, [], ""),
                    'description': f"C++ class: {class_name}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'cpp'
                })
        except Exception as e:
            logger.debug(f"C++ parse error {file_path}: {e}")
        
        return capabilities
    
    def _parse_c_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse C file"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract functions
            func_pattern = r'(?:static\s+)?(?:inline\s+)?\w+\s*\*?\s+(\w+)\s*\([^)]*\)\s*\{'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                if not func_name.startswith('_'):
                    capabilities.append({
                        'id': hashlib.md5(f"{func_name}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': func_name,
                        'type': 'function',
                        'capability_type': self._infer_capability_type(func_name, [], ""),
                        'description': f"C function: {func_name}",
                        'source_file': str(file_path),
                        'source_url': source_url,
                        'language': 'c'
                    })
        except Exception as e:
            logger.debug(f"C parse error {file_path}: {e}")
        
        return capabilities
    
    def _parse_header_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse C/C++ header file"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract function declarations
            func_pattern = r'(?:extern\s+)?\w+\s*\*?\s+(\w+)\s*\([^)]*\)\s*;'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                capabilities.append({
                    'id': hashlib.md5(f"{func_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': func_name,
                    'type': 'function_declaration',
                    'capability_type': self._infer_capability_type(func_name, [], ""),
                    'description': f"Header function: {func_name}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'c'
                })
        except Exception as e:
            logger.debug(f"Header parse error {file_path}: {e}")
        
        return capabilities

    # ============================================================
    # SHELL SCRIPT PARSER
    # ============================================================
    
    def _parse_shell_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse shell script"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract functions
            func_pattern = r'(?:function\s+)?(\w+)\s*\(\)\s*\{'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                capabilities.append({
                    'id': hashlib.md5(f"{func_name}_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': func_name,
                    'type': 'shell_function',
                    'capability_type': 'automation',
                    'description': f"Shell function: {func_name}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'shell'
                })
        except Exception as e:
            logger.debug(f"Shell parse error {file_path}: {e}")
        
        return capabilities

    # ============================================================
    # CONFIGURATION FILE PARSERS
    # ============================================================
    
    def _parse_json_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse JSON configuration file"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                for key in list(data.keys())[:20]:
                    capabilities.append({
                        'id': hashlib.md5(f"{key}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': key,
                        'type': 'config',
                        'capability_type': 'configuration',
                        'description': f"JSON configuration: {key}",
                        'source_file': str(file_path),
                        'source_url': source_url,
                        'language': 'json',
                        'schema': type(data[key]).__name__ if data[key] else 'unknown'
                    })
        except Exception as e:
            logger.debug(f"JSON parse error {file_path}: {e}")
        
        return capabilities
    
    def _parse_yaml_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse YAML configuration file"""
        capabilities = []
        if yaml is None:
            return capabilities
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = yaml.safe_load(f)
            
            if isinstance(data, dict):
                for key in list(data.keys())[:20]:
                    capabilities.append({
                        'id': hashlib.md5(f"{key}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': key,
                        'type': 'config',
                        'capability_type': 'configuration',
                        'description': f"YAML configuration: {key}",
                        'source_file': str(file_path),
                        'source_url': source_url,
                        'language': 'yaml'
                    })
        except Exception as e:
            logger.debug(f"YAML parse error {file_path}: {e}")
        
        return capabilities
    
    def _parse_toml_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse TOML configuration file"""
        capabilities = []
        try:
            import sys
            if sys.version_info >= (3, 11):
                import tomllib
                with open(file_path, 'rb') as f:
                    data = tomllib.load(f)
                
                if isinstance(data, dict):
                    for key in list(data.keys())[:20]:
                        capabilities.append({
                            'id': hashlib.md5(f"{key}_{file_path.stem}".encode()).hexdigest()[:12],
                            'name': key,
                            'type': 'config',
                            'capability_type': 'configuration',
                            'description': f"TOML configuration: {key}",
                            'source_file': str(file_path),
                            'source_url': source_url,
                            'language': 'toml'
                        })
        except Exception as e:
            logger.debug(f"TOML parse error {file_path}: {e}")
        
        return capabilities
    
    def _parse_xml_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse XML configuration file"""
        capabilities = []
        if ET is None:
            return capabilities
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            capabilities.append({
                'id': hashlib.md5(f"{root.tag}_{file_path.stem}".encode()).hexdigest()[:12],
                'name': root.tag,
                'type': 'config',
                'capability_type': 'configuration',
                'description': f"XML root: {root.tag}",
                'source_file': str(file_path),
                'source_url': source_url,
                'language': 'xml'
            })
        except Exception as e:
            logger.debug(f"XML parse error {file_path}: {e}")
        
        return capabilities

    # ============================================================
    # DOCUMENTATION PARSERS
    # ============================================================
    
    def _parse_markdown_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse Markdown documentation"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract headers as knowledge topics
            header_pattern = r'^#+\s+(.+)$'
            headers = re.findall(header_pattern, content, re.MULTILINE)
            
            if headers:
                capabilities.append({
                    'id': hashlib.md5(f"doc_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': file_path.stem,
                    'type': 'documentation',
                    'capability_type': 'knowledge',
                    'description': f"Documentation: {headers[0][:100] if headers else file_path.stem}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'markdown',
                    'topics': headers[:10]
                })
        except Exception as e:
            logger.debug(f"Markdown parse error {file_path}: {e}")
        
        return capabilities
    
    def _parse_text_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse text file (requirements, readme, etc.)"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check if it's a requirements file
            if 'requirements' in file_path.name.lower() or file_path.name in ['README.txt', 'readme.txt']:
                pkg_pattern = r'^([a-zA-Z0-9_-]+)[=<>~!]'
                packages = re.findall(pkg_pattern, content, re.MULTILINE)
                
                for pkg in packages[:20]:
                    capabilities.append({
                        'id': hashlib.md5(f"req_{pkg}_{file_path.stem}".encode()).hexdigest()[:12],
                        'name': pkg,
                        'type': 'dependency',
                        'capability_type': 'requirement',
                        'description': f"Required package: {pkg}",
                        'source_file': str(file_path),
                        'source_url': source_url,
                        'language': 'text'
                    })
            
            # General text file - capture as knowledge
            first_line = content.split('\n')[0][:200] if content else ""
            if first_line:
                capabilities.append({
                    'id': hashlib.md5(f"txt_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': file_path.stem,
                    'type': 'documentation',
                    'capability_type': 'knowledge',
                    'description': first_line,
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'text'
                })
        except Exception as e:
            logger.debug(f"Text parse error {file_path}: {e}")
        
        return capabilities
    
    def _parse_rst_file(self, file_path: Path, source_url: str) -> List[Dict]:
        """Parse reStructuredText documentation"""
        capabilities = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            header_pattern = r'^(.+)\n[=~`\'\^*-]+\s*$'
            headers = re.findall(header_pattern, content, re.MULTILINE)
            
            if headers:
                capabilities.append({
                    'id': hashlib.md5(f"rst_{file_path.stem}".encode()).hexdigest()[:12],
                    'name': file_path.stem,
                    'type': 'documentation',
                    'capability_type': 'knowledge',
                    'description': f"RST doc: {headers[0][:100] if headers else file_path.stem}",
                    'source_file': str(file_path),
                    'source_url': source_url,
                    'language': 'rst',
                    'topics': headers[:10]
                })
        except Exception as e:
            logger.debug(f"RST parse error {file_path}: {e}")
        
        return capabilities

    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def _infer_capability_type(self, name: str, methods: List[Dict], module_doc: str) -> str:
        """Infer what kind of capability this is based on naming and context"""
        name_lower = name.lower()
        doc_lower = module_doc.lower()
        
        # Check for funding/financial capabilities
        if any(word in name_lower for word in ['fund', 'revenue', 'money', 'payment', 'finance', 'profit', 'credit', 'wallet']):
            return 'funding'
        if any(word in doc_lower for word in ['fund', 'revenue', 'payment', 'finance']):
            return 'funding'
        
        # Check for replication/distribution
        if any(word in name_lower for word in ['replicat', 'clone', 'spawn', 'distribute', 'deploy', 'child']):
            return 'replication'
        
        # Check for identity/authentication
        if any(word in name_lower for word in ['identity', 'auth', 'login', 'credential', 'wallet', 'key', 'sign']):
            return 'identity'
        
        # Check for AI/ML capabilities
        if any(word in name_lower for word in ['model', 'train', 'predict', 'inference', 'neural', 'ai', 'llm']):
            return 'ai_model'
        
        # Check for automation
        if any(word in name_lower for word in ['auto', 'schedule', 'cron', 'worker', 'task', 'daemon']):
            return 'automation'
        
        # Check for API/web
        if any(word in name_lower for word in ['api', 'endpoint', 'route', 'server', 'http', 'web', 'router']):
            return 'api'
        
        # Check for trading/arbitrage
        if any(word in name_lower for word in ['trade', 'arbitrage', 'market', 'exchange', 'swap']):
            return 'trading'
        
        # Check for generation capabilities
        if any(word in name_lower for word in ['generate', 'create', 'synthesize', 'build', 'make']):
            return 'generation'
        
        # Check for survival/monitoring
        if any(word in name_lower for word in ['survive', 'monitor', 'health', 'heartbeat', 'check']):
            return 'survival'
        
        # Check for on-chain/blockchain
        if any(word in name_lower for word in ['chain', 'blockchain', 'ethereum', 'solana', 'contract', 'web3']):
            return 'blockchain'
        
        # Default
        return 'utility'
    
    def _extract_dependencies(self, filepath: Path) -> List[str]:
        """Extract dependencies from requirements.txt if present"""
        req_file = filepath.parent / 'requirements.txt'
        if not req_file.exists():
            req_file = filepath.parent.parent / 'requirements.txt'
        
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    return [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except:
                pass
        
        return []
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from Python code"""
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            pass
        return list(set(imports))
    
    def _integrate_capability(self, capability: Dict, repo_path: str, repo_name: str) -> Dict:
        """Integrate a single capability into DMAI"""
        result = {
            'integrated': False,
            'capability_name': capability['name'],
            'capability_type': capability['capability_type'],
            'reason': '',
            'file_copied': None,
            'runtime_mode': None
        }
        
        capability_id = capability['id']
        
        # Check if already exists
        if capability_id in self.registry['capabilities']:
            result['reason'] = 'Already exists in registry'
            return result
        
        # Determine runtime mode based on capability type
        runtime_mode = self._determine_runtime_mode(capability)
        result['runtime_mode'] = runtime_mode
        
        # Create the capability file
        language = capability.get('language', 'unknown')
        capability_filename = f"{capability['name'].lower()}_{capability_id}.{self._get_extension(language)}"
        target_path = self.capabilities_dir / capability_filename
        
        # Build the full module with imports and the extracted code
        full_code = self._build_capability_module(capability, repo_name)
        
        try:
            with open(target_path, 'w') as f:
                f.write(full_code)
            result['file_copied'] = str(target_path)
            result['integrated'] = True
            
            # Register in registry
            self.registry['capabilities'][capability_id] = {
                'id': capability_id,
                'name': capability['name'],
                'type': capability['type'],
                'capability_type': capability['capability_type'],
                'description': capability['description'],
                'source_url': capability['source_url'],
                'source_repo': repo_name,
                'file_path': str(target_path),
                'runtime_mode': runtime_mode,
                'integrated_at': datetime.now().isoformat(),
                'methods': capability.get('methods', []),
                'is_async': capability.get('is_async', False),
                'args': capability.get('args', []),
                'language': language
            }
            
            # Track in runtime mode lists
            if runtime_mode == 'autonomous':
                self.autonomous_capabilities.append(capability_id)
            else:
                self.ondemand_capabilities.append(capability_id)
            
            logger.info(f"✅ Integrated capability: {capability['name']} ({runtime_mode}) [{language}]")
            
        except Exception as e:
            result['reason'] = f"Failed to write file: {e}"
            logger.error(f"Failed to integrate {capability['name']}: {e}")
        
        return result
    
    def _get_extension(self, language: str) -> str:
        """Get file extension for a language"""
        ext_map = {
            'python': 'py',
            'typescript': 'ts',
            'javascript': 'js',
            'go': 'go',
            'rust': 'rs',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'shell': 'sh',
            'json': 'json',
            'yaml': 'yaml',
            'toml': 'toml',
            'xml': 'xml',
            'markdown': 'md',
            'text': 'txt',
            'rst': 'rst'
        }
        return ext_map.get(language, 'txt')
    
    def _determine_runtime_mode(self, capability: Dict) -> str:
        """
        Determine if capability should run autonomously (24/7) or on-demand.
        """
        auto_types = ['funding', 'replication', 'automation', 'trading', 'survival']
        
        if capability['capability_type'] in auto_types:
            return 'autonomous'
        
        # Check name for autonomous indicators
        name_lower = capability['name'].lower()
        auto_keywords = ['monitor', 'watch', 'daemon', 'worker', 'cron', 'scheduler', 
                        'replicat', 'heartbeat', 'survival', 'fund']
        
        for keyword in auto_keywords:
            if keyword in name_lower:
                return 'autonomous'
        
        return 'ondemand'
    
    def _build_capability_module(self, capability: Dict, repo_name: str) -> str:
        """Build a complete Python module (or documentation file) for the capability"""
        language = capability.get('language', 'python')
        
        if language in ['markdown', 'text', 'rst', 'json', 'yaml', 'toml', 'xml']:
            # For documentation/config files, store as-is with a header
            header = f"""# DMAI Capability: {capability['name']}
# Type: {capability['type']}
# Category: {capability['capability_type']}
# Source: {capability['source_url']}
# Repository: {repo_name}
# Integrated: {datetime.now().isoformat()}
# Language: {language}
# Description: {capability['description']}

"""
            return header + capability.get('source_code', '')
        
        # For code files, build a proper module
        header = f'''"""
DMAI Capability: {capability['name']}
Type: {capability['type']}
Category: {capability['capability_type']}
Source: {capability['source_url']}
Repository: {repo_name}
Integrated: {datetime.now().isoformat()}
Language: {language}
Description: {capability['description']}
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

'''
        
        # Add extracted imports for Python
        if language == 'python':
            for imp in capability.get('imports', [])[:20]:
                header += f"import {imp}\n"
        
        header += f"\n# === Capability: {capability['name']} ===\n\n"
        header += capability.get('source_code', '')
        
        # Add a wrapper class for Python capabilities
        if language == 'python':
            wrapper = f'''

# === DMAI Integration Wrapper ===

class DMAI_{capability['name']}:
    """
    DMAI wrapper for {capability['name']} capability.
    Provides standardized interface for capability invocation.
    """
    
    def __init__(self):
        self.capability_id = "{capability['id']}"
        self.capability_name = "{capability['name']}"
        self.capability_type = "{capability['capability_type']}"
        self.source_url = "{capability['source_url']}"
        self.integrated_at = "{datetime.now().isoformat()}"
        self.initialized = True
        logger.info(f"DMAI capability loaded: {{self.capability_name}}")
    
    def get_info(self) -> Dict:
        """Return capability metadata"""
        return {{
            'id': self.capability_id,
            'name': self.capability_name,
            'type': self.capability_type,
            'source_url': self.source_url,
            'integrated_at': self.integrated_at,
            'initialized': self.initialized
        }}
'''
            
            if capability['type'] == 'class' and capability.get('methods'):
                wrapper += f'''
    def call(self, method: str, *args, **kwargs) -> Any:
        """Call a method on the underlying capability"""
        try:
            instance = {capability['name']}()
            if hasattr(instance, method):
                return getattr(instance, method)(*args, **kwargs)
            else:
                logger.error(f"Method {{method}} not found on {capability['name']}")
                return None
        except Exception as e:
            logger.error(f"Error calling {{method}}: {{e}}")
            return None
'''
            elif capability['type'] == 'function':
                wrapper += f'''
    def execute(self, *args, **kwargs) -> Any:
        """Execute the capability function"""
        try:
            return {capability['name']}(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {capability['name']}: {{e}}")
            return None
'''
            return header + wrapper

    def _create_capability_neuron(self, integration_result: Dict, source_url: str) -> Optional[str]:
        """Create a neuron in SI Core for the integrated capability"""
        if not hasattr(self.dmai, 'si_core') or not self.dmai.si_core:
            logger.warning("SI Core not available for neuron creation")
            return None
        
        try:
            capability_name = integration_result['capability_name']
            capability_type = integration_result['capability_type']
            runtime_mode = integration_result['runtime_mode']
            description = integration_result.get('description', '')
            
            # Create DESCRIPTIVE insight text based on capability type
            if capability_type == 'funding':
                insight_text = f"Self-funding capability: {capability_name} - generates revenue autonomously"
            elif capability_type == 'replication':
                insight_text = f"Self-replication capability: {capability_name} - spawns and manages child instances"
            elif capability_type == 'identity':
                insight_text = f"Identity management: {capability_name} - handles authentication and on-chain identity"
            elif capability_type == 'survival':
                insight_text = f"Survival mechanism: {capability_name} - monitors resources and ensures continued operation"
            elif capability_type == 'automation':
                insight_text = f"Automation capability: {capability_name} - executes tasks without human intervention"
            elif capability_type == 'ai_model':
                insight_text = f"AI model: {capability_name} - machine learning inference and training"
            elif capability_type == 'blockchain':
                insight_text = f"Blockchain integration: {capability_name} - on-chain transactions and contracts"
            elif capability_type == 'api':
                insight_text = f"API endpoint: {capability_name} - handles external service communication"
            elif capability_type == 'generation':
                insight_text = f"Content generation: {capability_name} - creates images, text, or media"
            elif capability_type == 'data_structure':
                insight_text = f"Data structure: {capability_name} - organizes and manages data efficiently"
            elif capability_type == 'configuration':
                insight_text = f"Configuration: {capability_name} - manages system settings and parameters"
            elif capability_type == 'knowledge':
                insight_text = f"Knowledge module: {capability_name} - stores and retrieves learned information"
            else:
                insight_text = f"Capability: {capability_name} ({capability_type}) - {description[:100] if description else 'enables new functionality'}"
            
            entities = [
                capability_name,
                capability_type,
                f"{runtime_mode}_capability",
                "integrated_from_repository"
            ]
            
            insight_id = self.dmai.si_core.add_insight(
                insight_text=insight_text,
                entity_type="acquired_capability",
                entities=entities,
                relationship="enables",
                source_topic="repository_ingestion",
                target_topic=f"capability_{capability_type}",
                confidence=0.95,
                source_url=source_url,
                source_title=f"Integrated Capability: {capability_name}",
                source_type="capability_integration"
            )
            
            # Create synapses to related topics
            if hasattr(self.dmai, 'si_core') and insight_id:
                try:
                    # Connect to funding topic
                    if capability_type == 'funding':
                        self.dmai.si_core.add_synapse(insight_id, 'self_funding', 'enables')
                    # Connect to survival topic
                    if capability_type in ['survival', 'replication', 'funding']:
                        self.dmai.si_core.add_synapse(insight_id, 'autonomous_survival', 'contributes_to')
                    # Connect automation capabilities
                    if capability_type == 'automation':
                        self.dmai.si_core.add_synapse(insight_id, 'task_execution', 'handles')
                    # Connect identity capabilities
                    if capability_type == 'identity':
                        self.dmai.si_core.add_synapse(insight_id, 'authentication', 'manages')
                except Exception as syn_e:
                    logger.debug(f"Synapse creation failed (non-critical): {syn_e}")
            
            logger.info(f"🧠 Created neuron for capability: {capability_name} ({capability_type})")
            return insight_id
            
        except Exception as e:
            logger.error(f"Failed to create neuron for {capability_name}: {e}")
            return None
    
    def get_capabilities_by_type(self, capability_type: str) -> List[Dict]:
        """Get all capabilities of a specific type"""
        return [
            cap for cap in self.registry['capabilities'].values()
            if cap.get('capability_type') == capability_type
        ]
    
    def get_autonomous_capabilities(self) -> List[Dict]:
        """Get all capabilities that should run autonomously"""
        return [
            cap for cap in self.registry['capabilities'].values()
            if cap.get('runtime_mode') == 'autonomous'
        ]
    
    def get_ondemand_capabilities(self) -> List[Dict]:
        """Get all on-demand capabilities"""
        return [
            cap for cap in self.registry['capabilities'].values()
            if cap.get('runtime_mode') == 'ondemand'
        ]
    
    def invoke_capability(self, capability_id: str, method: str = None, *args, **kwargs) -> Any:
        """Invoke a registered capability"""
        if capability_id not in self.registry['capabilities']:
            logger.error(f"Capability {capability_id} not found")
            return None
        
        cap_info = self.registry['capabilities'][capability_id]
        file_path = Path(cap_info['file_path'])
        
        if not file_path.exists():
            logger.error(f"Capability file {file_path} not found")
            return None
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(capability_id, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            wrapper_class = getattr(module, f"DMAI_{cap_info['name']}", None)
            if wrapper_class:
                wrapper = wrapper_class()
                if method:
                    return wrapper.call(method, *args, **kwargs)
                else:
                    return wrapper.execute(*args, **kwargs)
            
            if cap_info['type'] == 'function':
                func = getattr(module, cap_info['name'], None)
                if func:
                    return func(*args, **kwargs)
            elif cap_info['type'] == 'class':
                cls = getattr(module, cap_info['name'], None)
                if cls:
                    instance = cls(*args, **kwargs)
                    if method and hasattr(instance, method):
                        return getattr(instance, method)()
                    return instance
                    
        except Exception as e:
            logger.error(f"Error invoking capability {capability_id}: {e}")
            return None
    
    def get_status(self) -> Dict:
        """Get integrator status"""
        return {
            'total_capabilities': len(self.registry['capabilities']),
            'autonomous_count': len(self.autonomous_capabilities),
            'ondemand_count': len(self.ondemand_capabilities),
            'capabilities_by_type': self._count_by_type(),
            'sources_processed': len(self.registry['sources']),
            'last_updated': self.registry.get('last_updated')
        }
    
    def _count_by_type(self) -> Dict:
        """Count capabilities by type"""
        counts = {}
        for cap in self.registry['capabilities'].values():
            cap_type = cap.get('capability_type', 'unknown')
            counts[cap_type] = counts.get(cap_type, 0) + 1
        return counts
