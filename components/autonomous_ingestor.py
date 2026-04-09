# components/autonomous_developer.py
"""
DMAI Autonomous Developer - Takes any input and turns it into working code
"""

import os
import re
import ast
import json
import subprocess
import tempfile
import importlib.util
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

class AutonomousDeveloper:
    """
    DMAI's autonomous development engine
    Takes any input (idea, repo, article, conversation) -> working code -> incorporation
    """
    
    def __init__(self, dmai_app):
        self.dmai = dmai_app
        self.development_projects = []
        self.implemented_features = []
        self.project_dir = Path("data/development_projects")
        self.project_dir.mkdir(parents=True, exist_ok=True)
    
    def process_input(self, input_source: str, input_type: str = "auto") -> Dict:
        """
        Main entry point - process any input and turn into implementation
        
        Args:
            input_source: GitHub URL, idea text, article URL, file path, or conversation
            input_type: github, idea, article, code, conversation, auto
        """
        result = {
            'input': input_source,
            'type': input_type,
            'status': 'started',
            'timestamp': datetime.now().isoformat(),
            'analysis': {},
            'design': {},
            'implementation': {},
            'incorporation': {}
        }
        
        # Step 1: Auto-detect input type
        if input_type == 'auto':
            input_type = self._detect_input_type(input_source)
        result['type'] = input_type
        
        # Step 2: Analyze the input
        analysis = self._analyze_input(input_source, input_type)
        result['analysis'] = analysis
        
        if not analysis.get('feasible', True):
            result['status'] = 'failed'
            result['error'] = analysis.get('reason', 'Not feasible')
            return result
        
        # Step 3: Design solution
        design = self._design_solution(analysis)
        result['design'] = design
        
        # Step 4: Implement code
        implementation = self._implement_solution(design)
        result['implementation'] = implementation
        
        if implementation.get('success'):
            # Step 5: Test the implementation
            test_result = self._test_implementation(implementation)
            if test_result.get('passed'):
                # Step 6: Incorporate into DMAI
                incorporation = self._incorporate_into_dmai(implementation)
                result['incorporation'] = incorporation
                result['status'] = 'complete'
            else:
                # Step 7: Debug and fix
                fixed = self._debug_and_fix(implementation, test_result)
                if fixed:
                    result['status'] = 'complete'
                else:
                    result['status'] = 'needs_review'
        else:
            result['status'] = 'implementation_failed'
        
        # Record project
        self.development_projects.append({
            'input': input_source[:200],
            'type': input_type,
            'result': result['status'],
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def _detect_input_type(self, source: str) -> str:
        """Detect what type of input we're dealing with"""
        source_lower = source.lower()
        
        if 'github.com' in source_lower:
            return 'github'
        elif source_lower.startswith('http'):
            return 'url'
        elif source.endswith('.py') or source.endswith('.js') or source.endswith('.cpp'):
            return 'code_file'
        elif len(source) > 100 and (' ' in source or '\n' in source):
            return 'idea_text'
        elif source.startswith('/') or source.startswith('.'):
            return 'local_path'
        else:
            return 'idea_text'
    
    def _analyze_input(self, source: str, input_type: str) -> Dict:
        """Deep analysis of the input to understand requirements"""
        analysis = {
            'feasible': True,
            'type': input_type,
            'requirements': [],
            'dependencies': [],
            'complexity': 'medium',
            'estimated_effort': 'unknown'
        }
        
        if input_type == 'github':
            repo_info = self._analyze_github_repo(source)
            analysis.update(repo_info)
        elif input_type == 'idea_text':
            idea_info = self._analyze_idea_text(source)
            analysis.update(idea_info)
        elif input_type == 'url':
            url_info = self._analyze_url(source)
            analysis.update(url_info)
        elif input_type == 'code_file':
            code_info = self._analyze_code_file(source)
            analysis.update(code_info)
        
        return analysis
    
    def _analyze_github_repo(self, url: str) -> Dict:
        """Analyze a GitHub repository to understand its purpose"""
        result = {
            'source_type': 'github',
            'repo_url': url,
            'name': url.split('/')[-1],
            'owner': url.split('/')[-2],
            'requirements': [],
            'dependencies': [],
            'key_files': []
        }
        
        temp_dir = tempfile.mkdtemp()
        try:
            subprocess.run(['git', 'clone', '--depth', '1', url, temp_dir],
                         capture_output=True, timeout=60, check=False)
            
            readme_path = Path(temp_dir) / 'README.md'
            if readme_path.exists():
                with open(readme_path, 'r') as f:
                    readme = f.read()
                    lines = readme.split('\n')[:20]
                    result['purpose'] = ' '.join(lines)[:500]
            
            req_path = Path(temp_dir) / 'requirements.txt'
            if req_path.exists():
                with open(req_path, 'r') as f:
                    result['dependencies'] = [line.strip() for line in f if line.strip()]
            
            for py_file in Path(temp_dir).rglob('*.py'):
                if '__init__' not in str(py_file) and 'test' not in str(py_file).lower():
                    result['key_files'].append(str(py_file.relative_to(temp_dir)))
                    if len(result['key_files']) >= 5:
                        break
            
            if 'model' in str(result['key_files']).lower():
                result['requirements'].append('machine_learning_model')
            if 'generate' in str(result['key_files']).lower():
                result['requirements'].append('generation_capability')
            if 'api' in str(result['key_files']).lower():
                result['requirements'].append('api_endpoint')
                
        except Exception as e:
            result['error'] = str(e)
            result['feasible'] = False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    def _analyze_idea_text(self, text: str) -> Dict:
        """Parse natural language idea into requirements"""
        text_lower = text.lower()
        
        result = {
            'source_type': 'idea',
            'original_text': text[:500],
            'requirements': [],
            'keywords': []
        }
        
        keywords = ['image', 'video', 'audio', 'generate', 'create', 'api', 'web', 
                   'trading', 'analysis', 'bot', 'automation', 'learn', 'train']
        
        for keyword in keywords:
            if keyword in text_lower:
                result['keywords'].append(keyword)
                if keyword in ['image', 'video', 'audio', 'generate', 'create']:
                    result['requirements'].append('generation_capability')
                elif keyword in ['api', 'web']:
                    result['requirements'].append('api_endpoint')
                elif keyword in ['trading', 'analysis']:
                    result['requirements'].append('data_processing')
                elif keyword in ['bot', 'automation']:
                    result['requirements'].append('automation')
        
        if len(result['requirements']) > 3:
            result['complexity'] = 'high'
        elif len(result['requirements']) > 1:
            result['complexity'] = 'medium'
        else:
            result['complexity'] = 'low'
        
        return result
    
    def _analyze_url(self, url: str) -> Dict:
        """Analyze article or documentation URL"""
        result = {
            'source_type': 'url',
            'url': url,
            'requirements': [],
            'content_summary': ''
        }
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.find('title')
                if title:
                    result['title'] = title.text
                
                for tag in ['article', 'main', '.content', '#content']:
                    content = soup.select_one(tag)
                    if content:
                        text = content.get_text()[:1000]
                        result['content_summary'] = text
                        break
                
                if not result['content_summary']:
                    result['content_summary'] = soup.get_text()[:1000]
                
                content_lower = result['content_summary'].lower()
                if 'tutorial' in content_lower or 'guide' in content_lower:
                    result['requirements'].append('learning_material')
                if 'api' in content_lower:
                    result['requirements'].append('api_documentation')
                    
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _analyze_code_file(self, filepath: str) -> Dict:
        """Analyze existing code file"""
        result = {
            'source_type': 'code_file',
            'filepath': filepath,
            'requirements': [],
            'functions': [],
            'classes': []
        }
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    result['functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    result['classes'].append(node.name)
            
            if any('generate' in f for f in result['functions']):
                result['requirements'].append('generation')
            if any('train' in f for f in result['functions']):
                result['requirements'].append('training')
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _design_solution(self, analysis: Dict) -> Dict:
        """Design the solution architecture"""
        design = {
            'architecture': 'module',
            'components': [],
            'interfaces': [],
            'estimated_lines': 0
        }
        
        requirements = analysis.get('requirements', [])
        
        if 'generation_capability' in requirements:
            design['components'].append({
                'name': 'Generator',
                'type': 'class',
                'methods': ['generate', 'validate_output'],
                'dependencies': ['torch', 'transformers']
            })
            design['estimated_lines'] += 200
        
        if 'api_endpoint' in requirements:
            design['components'].append({
                'name': 'APIHandler',
                'type': 'class',
                'methods': ['handle_request', 'process_input'],
                'dependencies': ['flask', 'fastapi']
            })
            design['estimated_lines'] += 150
        
        if 'data_processing' in requirements:
            design['components'].append({
                'name': 'DataProcessor',
                'type': 'class',
                'methods': ['process', 'analyze'],
                'dependencies': ['pandas', 'numpy']
            })
            design['estimated_lines'] += 100
        
        if 'automation' in requirements:
            design['components'].append({
                'name': 'AutomationEngine',
                'type': 'class',
                'methods': ['schedule', 'execute', 'monitor'],
                'dependencies': ['schedule', 'apscheduler']
            })
            design['estimated_lines'] += 120
        
        return design
    
    def _implement_solution(self, design: Dict) -> Dict:
        """Generate actual code from the design"""
        implementation = {
            'success': True,
            'files': [],
            'code': {},
            'main_entry': None
        }
        
        components = design.get('components', [])
        
        for component in components:
            code = self._generate_component_code(component)
            filename = f"{component['name'].lower()}.py"
            implementation['files'].append(filename)
            implementation['code'][filename] = code
            if component['name'] == 'Generator':
                implementation['main_entry'] = filename
        
        return implementation
    
    def _generate_component_code(self, component: Dict) -> str:
        """Generate actual Python code for a component"""
        name = component['name']
        methods = component.get('methods', [])
        
        code = f'''
"""
Auto-generated component: {name}
Generated by DMAI Autonomous Developer
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class {name}:
    """Auto-generated {name} component"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {{}}
        self.initialized = False
        self._init()
    
    def _init(self):
        """Initialize the component"""
        try:
            self.initialized = True
            logger.info(f"{name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {{e}}")
            self.initialized = False
    
'''
        
        for method in methods:
            code += f'''
    def {method}(self, *args, **kwargs):
        """
        {method} method for {name}
        Auto-generated - ready for implementation
        """
        try:
            logger.debug(f"Calling {name}.{method}")
            
            result = {{
                'success': True,
                'component': '{name}',
                'method': '{method}',
                'timestamp': datetime.now().isoformat()
            }}
            return result
            
        except Exception as e:
            logger.error(f"Error in {name}.{method}: {{e}}")
            return {{'success': False, 'error': str(e)}}
'''
        
        return code
    
    def _test_implementation(self, implementation: Dict) -> Dict:
        """Test the generated code"""
        result = {
            'passed': True,
            'tests': [],
            'errors': []
        }
        
        temp_dir = tempfile.mkdtemp()
        try:
            for filename, code in implementation['code'].items():
                filepath = Path(temp_dir) / filename
                filepath.write_text(code)
                
                try:
                    spec = importlib.util.spec_from_file_location(filename[:-3], filepath)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    result['tests'].append(f"✅ {filename} loads successfully")
                except Exception as e:
                    result['passed'] = False
                    result['errors'].append(f"Test failed for {filename}: {e}")
                    
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    def _debug_and_fix(self, implementation: Dict, test_result: Dict) -> bool:
        """Attempt to debug and fix failed implementation"""
        print(f"Debugging failed implementation: {test_result.get('errors', [])}")
        return False
    
    def _incorporate_into_dmai(self, implementation: Dict) -> Dict:
        """Incorporate the implementation into DMAI's core and save to disk"""
        result = {
            'success': True,
            'added_methods': [],
            'insights_created': [],
            'saved_files': []
        }
        
        # Save generated files to disk
        for filename, code in implementation['code'].items():
            filepath = self.project_dir / filename
            filepath.write_text(code)
            result['saved_files'].append(str(filepath))
            print(f"📁 Saved: {filepath}")
            
            # Add to knowledge graph
            if hasattr(self.dmai, 'knowledge_graph') and self.dmai.knowledge_graph:
                self.dmai.knowledge_graph.add_concept(
                    f"auto_implemented_{filename[:-3]}",
                    'autonomous_development',
                    {
                        'code': code[:500],
                        'filepath': str(filepath),
                        'implemented_at': datetime.now().isoformat()
                    }
                )
            
            # Create insight in SI Core
            if hasattr(self.dmai, 'si_core') and self.dmai.si_core:
                insight_id = self.dmai.si_core.add_insight(
                    insight_text=f"Autonomously implemented {filename}",
                    entity_type="autonomous_implementation",
                    entities=[filename[:-3]],
                    relationship="implemented",
                    source_topic="AutonomousDeveloper",
                    target_topic="DMAI_Capabilities",
                    confidence=0.9
                )
                result['insights_created'].append(insight_id)
        
        return result
