"""
Data Validator - Ensures all generated code uses REAL data, not placeholders
Prevents the system from generating fake/hardcoded data unless explicitly requested
"""

import ast
import re
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - VALIDATOR - %(message)s')

class DataValidator:
    def __init__(self):
        self.forbidden_patterns = [
            # Random/fake data generators
            r'random\.randint\(.*\)',
            r'np\.random\.rand.*\(.*\)',
            r'torch\.rand.*\(.*\)',
            r'generate_fake_data\(',
            r'create_mock_data\(',
            r'synthetic_data\(',
            r'fake_data\(\)',
            r'mock_data\(\)',
            
            # Hardcoded placeholder data
            r'data\s*=\s*\[\s*\]',
            r'data\s*=\s*\{\s*\}',
            r'\[\s*1\s*,\s*2\s*,\s*3\s*,\s*4\s*,\s*5\s*\]',  # Simple number arrays
            r'\{\s*"id"\s*:\s*\d+\s*,\s*"name"\s*:\s*"[^"]*"\s*\}',  # Hardcoded dicts
            
            # Placeholder values
            r'placeholder',
            r'dummy_data',
            r'test_data',
            r'sample_data',
            r'example_data',
            
            # API key placeholders
            r'api_key\s*=\s*"YOUR_API_KEY"',
            r'api_secret\s*=\s*"YOUR_SECRET"',
            r'password\s*=\s*"password"',
            r'secret\s*=\s*"secret"',
            r'token\s*=\s*"YOUR_TOKEN"',
        ]
        
        self.real_data_sources = [
            'yfinance',
            'alpaca',
            'binance',
            'coinbase',
            'iex',
            'quandl',
            'fred',
            'econdb',
            'pandas_datareader',
            'sqlite3',
            'psycopg2',
            'pymongo',
            'requests.get(',
            'urllib.request',
            'websocket',
            'kafka',
            'rabbitmq',
            'redis',
            'api.get(',
            'client.get(',
            'fetch_data(',
            'load_data(',
            'pd.read_csv(',
            'pd.read_excel(',
            'pd.read_json(',
        ]
        
        self.user_requested_patterns = [
            'demo_mode',
            'synthesis',
            'example_data',
            'test_mode',
            'development',
            'debug_mode',
            'mock_mode',
            'simulation',
            'demo=True',
            'test=True',
        ]
    
    def validate_code(self, code, purpose="production"):
        """
        Validate that code uses real data sources
        Returns: (is_valid, issues, suggestions)
        """
        issues = []
        suggestions = []
        
        # Check if user explicitly requested fake data
        user_requested_fake = any(pattern in code.lower() for pattern in self.user_requested_patterns)
        
        # Check for forbidden patterns (if not user-requested)
        if not user_requested_fake:
            for pattern in self.forbidden_patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    issues.append({
                        'type': 'fake_data',
                        'pattern': pattern,
                        'match': match.group(),
                        'line': self._get_line_number(code, match.start())
                    })
        
        # Check for real data sources
        has_real_data = any(source in code for source in self.real_data_sources)
        
        if not has_real_data and not user_requested_fake and len(code.strip()) > 100:
            suggestions.append({
                'type': 'add_real_data',
                'message': 'No real data sources detected. Consider adding:',
                'sources': self.real_data_sources[:5]  # Suggest top 5
            })
        
        # Parse AST for deeper analysis
        try:
            tree = ast.parse(code)
            ast_issues = self._analyze_ast(tree)
            issues.extend(ast_issues)
        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'error': str(e)
            })
        
        return {
            'is_valid': len(issues) == 0 or user_requested_fake,
            'issues': issues,
            'suggestions': suggestions,
            'user_requested_fake': user_requested_fake
        }
    
    def _get_line_number(self, code, position):
        """Get line number from character position"""
        return code.count('\n', 0, position) + 1
    
    def _analyze_ast(self, tree):
        """Analyze AST for data patterns"""
        issues = []
        
        for node in ast.walk(tree):
            # Check for hardcoded lists
            if isinstance(node, ast.List) and len(node.elts) < 10:
                if all(isinstance(elt, ast.Constant) for elt in node.elts):
                    issues.append({
                        'type': 'hardcoded_list',
                        'line': node.lineno,
                        'message': f'Hardcoded list with {len(node.elts)} elements'
                    })
            
            # Check for empty dictionaries
            if isinstance(node, ast.Dict) and len(node.keys) == 0:
                issues.append({
                    'type': 'empty_dict',
                    'line': node.lineno,
                    'message': 'Empty dictionary - likely placeholder data'
                })
            
            # Check for assignments of literal values
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, (int, float)) and node.value.value in [0, 1, 100, 1000]:
                        issues.append({
                            'type': 'magic_number',
                            'line': node.lineno,
                            'message': f'Magic number {node.value.value} - consider using real data'
                        })
        
        return issues
    
    def suggest_real_data_source(self, domain):
        """Suggest real data sources based on domain"""
        data_sources = {
            'finance': ['yfinance', 'alpaca', 'binance', 'iex', 'quandl'],
            'stock': ['yfinance', 'alpaca', 'iex', 'polygon', 'tdameritrade'],
            'crypto': ['binance', 'coinbase', 'ccxt', 'cryptocompare', 'coingecko'],
            'weather': ['openweathermap', 'weather-api', 'noaa', 'weather-gov', 'meteostat'],
            'news': ['newsapi', 'gnews', 'mediastack', 'nytimes', 'guardian'],
            'social': ['tweepy', 'snscrape', 'facebook-sdk', 'instagram-api', 'reddit'],
            'database': ['sqlite3', 'psycopg2', 'pymongo', 'sqlalchemy', 'peewee'],
            'api': ['requests', 'httpx', 'aiohttp', 'urllib', 'grequests'],
            'csv': ['pandas', 'csv', 'openpyxl', 'xlrd'],
            'market': ['yfinance', 'alpaca', 'tdameritrade', 'robinhood', 'polygon']
        }
        
        for key, sources in data_sources.items():
            if key in domain.lower():
                return sources
        
        return self.real_data_sources
    
    def auto_fix(self, code, domain="general"):
        """Automatically fix common data validation issues"""
        lines = code.split('\n')
        fixed_lines = []
        
        # Get suggested real data sources
        suggested = self.suggest_real_data_source(domain)
        
        for line in lines:
            fixed_line = line
            
            # Replace placeholder API keys
            if 'api_key = "YOUR_API_KEY"' in line:
                fixed_line = line.replace(
                    'api_key = "YOUR_API_KEY"',
                    f'api_key = os.environ.get("API_KEY")  # Set in environment variables'
                )
            if 'api_secret = "YOUR_SECRET"' in line:
                fixed_line = line.replace(
                    'api_secret = "YOUR_SECRET"',
                    f'api_secret = os.environ.get("API_SECRET")'
                )
            
            # Replace random data generation with comments
            if 'random.randint' in line and '#' not in line:
                fixed_line = '# ' + line + f'\n# TODO: Replace with real data from {suggested[0] if suggested else "real data source"}'
            
            # Mark hardcoded arrays
            if ' = [' in line and '1,2,3,4,5' in line.replace(' ', ''):
                fixed_line = '# ' + line + f'\n# TODO: Load this data from {suggested[0] if suggested else "external source"}'
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)

if __name__ == "__main__":
    # Test the validator
    validator = DataValidator()
    
    # Test with bad code
    bad_code = """
def get_stock_data():
    # This is bad - uses fake data
    import random
    data = []
    for i in range(10):
        data.append({
            'price': random.randint(100, 200),
            'volume': 1000
        })
    return data
"""
    
    print("🔍 Testing bad code:")
    result = validator.validate_code(bad_code)
    print(json.dumps(result, indent=2))
    
    # Test with good code
    good_code = """
def get_stock_data():
    # This is good - uses real API
    import yfinance as yf
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1mo")
    return data
"""
    
    print("\n✅ Testing good code:")
    result = validator.validate_code(good_code)
    print(json.dumps(result, indent=2))
    
    # Test auto-fix
    print("\n🛠️ Auto-fix example:")
    fixed = validator.auto_fix(bad_code, "stocks")
    print(fixed)
