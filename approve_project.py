#!/usr/bin/env python3
"""
Approval script for Alex Riviera's projects
Run: python3 approve_project.py <project_id> approve
     python3 approve_project.py <project_id> reject "Notes here"
"""

import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 approve_project.py <project_id> approve")
        print("       python3 approve_project.py <project_id> reject \"Your notes\"")
        sys.exit(1)
    
    project_id = sys.argv[1]
    action = sys.argv[2]
    notes = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Load state
    state_file = Path("data/alex_projects/publishing_state.json")
    if not state_file.exists():
        print("❌ No publishing state found")
        sys.exit(1)
    
    with open(state_file) as f:
        state = json.load(f)
    
    # Find project
    found = False
    for i, approval in enumerate(state.get('pending_approvals', [])):
        if approval['id'] == project_id:
            found = True
            if action.lower() == 'approve':
                approval['status'] = 'approved'
                approval['approved_at'] = __import__('datetime').datetime.now().isoformat()
                state['approved_projects'].append(approval)
                state['pending_approvals'].pop(i)
                print(f"✅ Project '{approval['title']}' APPROVED")
            elif action.lower() == 'reject':
                approval['status'] = 'rejected'
                approval['rejected_at'] = __import__('datetime').datetime.now().isoformat()
                approval['rejection_notes'] = notes
                print(f"❌ Project '{approval['title']}' REJECTED")
                if notes:
                    print(f"   Notes: {notes}")
            break
    
    if not found:
        print(f"❌ Project '{project_id}' not found")
        sys.exit(1)
    
    # Save state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print("\n📁 Review files saved in: data/alex_projects/for_review/")

if __name__ == "__main__":
    main()
