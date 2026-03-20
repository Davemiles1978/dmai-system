#!/usr/bin/env python3
"""
Test Sanitizer - Removes special characters from test class names
Run this before building to ensure clean tests
"""

import os
import re
import glob

def sanitize_test_file(test_path):
    """Remove special characters from test class names"""
    with open(test_path, 'r') as f:
        content = f.read()
    
    # Replace # with _
    content = re.sub(r'class Test(\w*)#(\w*)\(unittest\.TestCase\):', 
                    r'class Test\1_\2(unittest.TestCase):', content)
    
    # Replace . with _
    content = re.sub(r'class Test(\w*)\.(\w*)\(unittest\.TestCase\):', 
                    r'class Test\1_\2(unittest.TestCase):', content)
    
    # Replace ( and ) with _
    content = re.sub(r'class Test(.*?)\((.*?)\)\(unittest\.TestCase\):', 
                    r'class Test\1_\2(unittest.TestCase):', content)
    
    # Generic catch-all: replace any remaining special characters
    content = re.sub(r'class Test(.*?)\(unittest\.TestCase\):', 
                    lambda m: f'class Test{re.sub(r"[^a-zA-Z0-9]", "_", m.group(1))}(unittest.TestCase):', 
                    content)
    
    with open(test_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Sanitized: {test_path}")

def main():
    test_files = glob.glob("tests/*_test.py")
    print(f"Found {len(test_files)} test files to sanitize...")
    
    for test_file in test_files:
        sanitize_test_file(test_file)
    
    print("\n🎉 All tests sanitized!")

if __name__ == "__main__":
    main()
