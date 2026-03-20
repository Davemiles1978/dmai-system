#!/usr/bin/env python3
"""Test script for key workflow system"""

from key_workflow import RawKey, KeyWorkflowOrchestrator
from datetime import datetime
import json

def main():
    print("🔑 Testing Key Workflow System")
    print("=" * 50)
    
    # Initialize workflow
    workflow = KeyWorkflowOrchestrator("workflow_config.json")
    
    # Test with various key types
    test_keys = [
        {
            'key': 'ghp_test123456789012345678901234567890123456',
            'type': 'github',
            'repo': 'test/repo1',
            'context': 'api_key = "ghp_test123456789012345678901234567890123456"'
        },
        {
            'key': 'AIzaSyDbydj-hg684EXca_rEwnBEW1AOQYhipPw',
            'type': 'google_api',
            'repo': 'test/repo2',
            'context': 'GOOGLE_API_KEY = "AIzaSyDbydj-hg684EXca_rEwnBEW1AOQYhipPw"'
        },
        {
            'repo': 'test/repo3',
        },
        {
            'key': '0aYAx2eY37dkfjqsrrZ53SSCkY1yY2kRYGvY27rv',
            'type': 'unknown',
            'repo': 'test/repo4',
            'context': 'SECRET_KEY = "0aYAx2eY37dkfjqsrrZ53SSCkY1yY2kRYGvY27rv"'
        }
    ]
    
    for test in test_keys:
        print(f"\n🔍 Testing {test['type']} key...")
        
        raw = RawKey(
            key_string=test['key'],
            source_repo=test['repo'],
            source_url=f"https://github.com/{test['repo']}/blob/main/file.py",
            source_file="file.py",
            line_number=10,
            context=test['context'],
            discovered_at=datetime.now()
        )
        
        result = workflow.process_raw_key(raw)
        
        if result:
            print(f"  ✅ Valid - Type: {result.identified_key.key_type}")
            print(f"     Weight: {result.weight}")
            print(f"     Message: {result.validation_message}")
        else:
            print(f"  ❌ Invalid or below threshold")
    
    # Show stats
    print("\n📊 Final Statistics:")
    stats = workflow.get_stats()
    print(json.dumps(stats, indent=2, default=str))
    
    # Show stored keys
    print("\n💾 Stored Keys:")
    keys = workflow.database.get_valid_keys(min_weight=5)
    for key in keys:
        print(f"  • {key['key_type']}: {key['key_preview']} (weight: {key['weight']})")

if __name__ == "__main__":
    main()
