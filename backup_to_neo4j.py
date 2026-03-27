#!/usr/bin/env python3
"""
Backup DMAI data to Neo4j
Run this to sync existing data to cloud storage
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/Users/davidmiles/Desktop/dmai-system')
from components.neo4j_storage import get_neo4j_storage

def main():
    print("☁️ DMAI Neo4j Backup Tool")
    print("=" * 40)
    
    # Initialize storage
    storage = get_neo4j_storage()
    
    # Load current local state
    data_path = Path('/Users/davidmiles/Desktop/dmai-system/data')
    
    # Evolution state
    evolution = {}
    if (data_path / 'evolution.json').exists():
        with open(data_path / 'evolution.json') as f:
            evolution = json.load(f)
        print(f"📊 Evolution: consciousness={evolution.get('consciousness', 0):.2%}, neurons={evolution.get('neurons', 0)}")
    
    # Persona
    persona = {}
    if (data_path / 'persona.json').exists():
        with open(data_path / 'persona.json') as f:
            persona = json.load(f)
        print(f"👤 Persona: style={persona.get('speaking_style', 'unknown')}")
    
    # Tasks
    tasks = []
    if (data_path / 'master_task.json').exists():
        with open(data_path / 'master_task.json') as f:
            tasks_data = json.load(f)
            if isinstance(tasks_data, dict) and 'tasks' in tasks_data:
                tasks = tasks_data['tasks']
            elif isinstance(tasks_data, list):
                tasks = tasks_data
        print(f"📋 Tasks: {len(tasks)} tasks")
    
    # Backup to Neo4j
    print("\n📦 Backing up to Neo4j...")
    
    storage.save_evolution_state({
        'consciousness': evolution.get('consciousness', 0),
        'neurons': evolution.get('neurons', 0),
        'synapses': evolution.get('synapses', 0),
        'evolution_cycles': evolution.get('evolution_cycles', 0),
        'evolution_count': evolution.get('evolution_count', 0)
    })
    
    storage.save_persona(persona)
    
    for task in tasks:
        storage.save_task(task)
    
    print("✅ Backup complete!")
    
    # Verify
    print("\n🔍 Verifying Neo4j data:")
    restored = storage.restore_all()
    
    if restored['evolution']:
        print(f"  ✅ Evolution: consciousness={restored['evolution'].get('consciousness', 0):.2%}, neurons={restored['evolution'].get('neurons', 0)}")
    if restored['persona']:
        print(f"  ✅ Persona: style={restored['persona'].get('speaking_style', 'unknown')}")
    if restored['tasks']:
        print(f"  ✅ Tasks: {len(restored['tasks'])} tasks")
    
    storage.close()
    print("\n🎉 Your data is now safely stored in Neo4j!")

if __name__ == '__main__':
    main()
