# components/capability_integrator.py
"""
DMAI Capability Integrator - Extracts and integrates actual capabilities from repositories
"""

import os
import ast
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class CapabilityIntegrator:
    """
    Extracts actual functions/classes from repositories and integrates them into DMAI.
    
    Unlike AutonomousDeveloper which generates placeholder stubs, this system:
    1. Clones and analyzes ALL Python files in a repo
    2. Extracts actual implemented classes and functions
    3. Compares against existing capabilities
    4. Integrates missing capabilities into components/capabilities/
    5. Registers them as callable functions
    6. Creates proper SI Core neurons for each capability
    7. Tracks runtime mode (autonomous 24/7 vs on-demand)
    """
    
    def __init__(self, dmai_app):
        self.dmai = dmai_app
        self.capabilities_dir = Path("components/capabilities")
        self.capabilities_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = Path("data/capabilities/registry.json")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.registry = self._load_registry()
        
        # Runtime mode tracking
        self.autonomous_capabilities = []  # Run 24/7
        self.ondemand_capabilities = []    # Run when requested
        
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
        
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
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
            
            # Step 2: Extract all capabilities from Python files
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
        """Extract all classes and functions from Python files in the repository"""
        capabilities = []
        
        for py_file in Path(repo_path).rglob('*.py'):
            # Skip test files, __init__.py, and virtual environments
            if 'test' in str(py_file).lower():
                continue
            if 'venv' in str(py_file) or 'env' in str(py_file):
                continue
            if '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Get module docstring for description
                module_doc = ast.get_docstring(tree) or ""
                
                # Extract all classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        capability = self._extract_class_capability(
                            node, py_file, content, module_doc, source_url
                        )
                        if capability:
                            capabilities.append(capability)
                    
                    elif isinstance(node, ast.FunctionDef):
                        # Only extract top-level functions (not methods inside classes)
                        if self._is_top_level_function(node, tree):
                            capability = self._extract_function_capability(
                                node, py_file, content, module_doc, source_url
                            )
                            if capability:
                                capabilities.append(capability)
                                
            except Exception as e:
                logger.warning(f"Could not parse {py_file}: {e}")
        
        return capabilities
    
    def _extract_class_capability(self, node: ast.ClassDef, filepath: Path, 
                                   content: str, module_doc: str, source_url: str) -> Optional[Dict]:
        """Extract a class as a capability"""
        class_name = node.name
        
        # Skip private classes and base classes that look abstract
        if class_name.startswith('_'):
            return None
        if 'Abstract' in class_name or 'Base' in class_name and len(node.bases) > 0:
            # Still include if it has concrete methods
            pass
        
        # Get class docstring
        docstring = ast.get_docstring(node) or f"Class {class_name} from {filepath.name}"
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if not item.name.startswith('_'):  # Skip private methods
                    methods.append({
                        'name': item.name,
                        'docstring': ast.get_docstring(item) or "",
                        'args': [arg.arg for arg in item.args.args],
                        'is_async': isinstance(item, ast.AsyncFunctionDef)
                    })
        
        # Get the actual source code for this class
        class_source = self._get_node_source(content, node)
        
        # Determine capability type from class name and methods
        capability_type = self._infer_capability_type(class_name, methods, module_doc)
        
        # Calculate a unique ID based on name and source
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
            'imports': self._extract_imports(content)
        }
    
    def _extract_function_capability(self, node: ast.FunctionDef, filepath: Path,
                                      content: str, module_doc: str, source_url: str) -> Optional[Dict]:
        """Extract a function as a capability"""
        func_name = node.name
        
        # Skip private functions
        if func_name.startswith('_'):
            return None
        
        # Get function docstring
        docstring = ast.get_docstring(node) or f"Function {func_name} from {filepath.name}"
        
        # Get the actual source code
        func_source = self._get_node_source(content, node)
        
        # Determine capability type
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
            'imports': self._extract_imports(content)
        }
    
    def _is_top_level_function(self, node: ast.FunctionDef, tree: ast.Module) -> bool:
        """Check if function is defined at module level (not inside a class)"""
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
    
    def _infer_capability_type(self, name: str, methods: List[Dict], module_doc: str) -> str:
        """Infer what kind of capability this is based on naming and context"""
        name_lower = name.lower()
        doc_lower = module_doc.lower()
        
        # Check for funding/financial capabilities
        if any(word in name_lower for word in ['fund', 'revenue', 'money', 'payment', 'finance', 'profit']):
            return 'funding'
        if any(word in doc_lower for word in ['fund', 'revenue', 'payment', 'finance']):
            return 'funding'
        
        # Check for replication/distribution
        if any(word in name_lower for word in ['replicat', 'clone', 'spawn', 'distribute', 'deploy']):
            return 'replication'
        
        # Check for identity/authentication
        if any(word in name_lower for word in ['identity', 'auth', 'login', 'credential', 'wallet']):
            return 'identity'
        
        # Check for AI/ML capabilities
        if any(word in name_lower for word in ['model', 'train', 'predict', 'inference', 'neural', 'ai']):
            return 'ai_model'
        
        # Check for automation
        if any(word in name_lower for word in ['auto', 'schedule', 'cron', 'worker', 'task']):
            return 'automation'
        
        # Check for API/web
        if any(word in name_lower for word in ['api', 'endpoint', 'route', 'server', 'http', 'web']):
            return 'api'
        
        # Check for trading/arbitrage
        if any(word in name_lower for word in ['trade', 'arbitrage', 'market', 'exchange', 'swap']):
            return 'trading'
        
        # Check for generation capabilities
        if any(word in name_lower for word in ['generate', 'create', 'synthesize', 'build']):
            return 'generation'
        
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
        """Extract import statements from code"""
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
        capability_filename = f"{capability['name'].lower()}_{capability_id}.py"
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
                'args': capability.get('args', [])
            }
            
            # Track in runtime mode lists
            if runtime_mode == 'autonomous':
                self.autonomous_capabilities.append(capability_id)
            else:
                self.ondemand_capabilities.append(capability_id)
            
            logger.info(f"✅ Integrated capability: {capability['name']} ({runtime_mode})")
            
        except Exception as e:
            result['reason'] = f"Failed to write file: {e}"
            logger.error(f"Failed to integrate {capability['name']}: {e}")
        
        return result
    
    def _determine_runtime_mode(self, capability: Dict) -> str:
        """
        Determine if capability should run autonomously (24/7) or on-demand.
        
        Autonomous: funding, replication, monitoring, continuous learning
        On-demand: generation, API endpoints, utilities
        """
        auto_types = ['funding', 'replication', 'automation', 'trading']
        
        if capability['capability_type'] in auto_types:
            return 'autonomous'
        
        # Check name for autonomous indicators
        name_lower = capability['name'].lower()
        auto_keywords = ['monitor', 'watch', 'daemon', 'worker', 'cron', 'scheduler', 'replicat']
        
        for keyword in auto_keywords:
            if keyword in name_lower:
                return 'autonomous'
        
        return 'ondemand'
    
    def _build_capability_module(self, capability: Dict, repo_name: str) -> str:
        """Build a complete Python module for the capability"""
        header = f'''"""
Capability: {capability['name']}
Type: {capability['type']}
Category: {capability['capability_type']}
Source: {capability['source_url']}
Repository: {repo_name}
Integrated: {datetime.now().isoformat()}
Description: {capability['description']}
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Original imports from source
'''
        
        # Add extracted imports
        for imp in capability.get('imports', [])[:20]:  # Limit imports
            header += f"import {imp}\n"
        
        header += f"\n# === Capability: {capability['name']} ===\n\n"
        header += capability['source_code']
        
        # Add a wrapper class for easy invocation
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
        
        # Add method wrappers if it's a class
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
            
            insight_text = f"Acquired capability: {capability_name} ({capability_type}) - runs {runtime_mode}"
            
            # Create entities based on capability
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
            
            logger.info(f"🧠 Created neuron for capability: {capability_name}")
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
            
            # Use the wrapper class
            wrapper_class = getattr(module, f"DMAI_{cap_info['name']}", None)
            if wrapper_class:
                wrapper = wrapper_class()
                if method:
                    return wrapper.call(method, *args, **kwargs)
                else:
                    return wrapper.execute(*args, **kwargs)
            
            # Fallback: try direct invocation
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
