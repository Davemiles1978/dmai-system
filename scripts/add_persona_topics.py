#!/usr/bin/env python3
"""Add persona topics to syllabus_topics.json"""

import json
from pathlib import Path
from datetime import datetime

PERSONA_TOPICS = [
    {
        "stage": "Toddler",
        "priority": 18,  # After Mandarin Chinese (17)
        "topic": "Cultural Knowledge Fundamentals",
        "category": "Core",
        "mastery": 2,
        "why_important": "Music, film, books, lifestyle references for authentic conversation as Alex Riviera (age 28)"
    },
    {
        "stage": "Child",
        "priority": 30,  # After Visual Storytelling (30? need to check)
        "topic": "Speech Pattern Integration",
        "category": "Core",
        "mastery": 3,
        "why_important": "Natural use of slang, idioms, fillers, and age-appropriate language patterns"
    },
    {
        "stage": "Teen",
        "priority": 38,  # After Interactive Art (37)
        "topic": "Persona Consistency & Evolution",
        "category": "Core",
        "mastery": 3,
        "why_important": "Maintaining authentic identity across all interactions while allowing natural growth"
    }
]

CATEGORY_COLORS = {
    "Core": "#4477ff",
    "Artistic": "#ff44cc",
    "Wealth": "#ffaa00",
    "Reverse": "#aa44ff",
    "Accelerator": "#00cc88",
}

def main():
    syllabus_path = Path(__file__).parent.parent / "data" / "syllabus_topics.json"
    
    with open(syllabus_path, 'r') as f:
        syllabus = json.load(f)
    
    print(f"Before: {len(syllabus['all_topics'])} topics, Core: {syllabus['topics_by_category'].get('Core', 0)}")
    
    for pt in PERSONA_TOPICS:
        topic_dict = {
            "id": f"topic_{pt['stage'].lower()}_{pt['priority']:02d}",
            "topic": pt['topic'],
            "category": pt['category'],
            "color": CATEGORY_COLORS[pt['category']],
            "stage": pt['stage'],
            "priority": pt['priority'],
            "mastery_required": pt['mastery'],
            "mastery_passes": pt['mastery'],
            "why_important": pt['why_important'],
            "status": "not_started",
            "progress": 0.0,
            "micro_neurons_created": 0,
            "synapses_created": 0,
            "last_updated": None,
            "continuous_learning": True,  # NEW: Flag for topics that require ongoing updates
            "last_researched": None       # NEW: Track when topic was last refreshed
        }
        
        syllabus['stages'][pt['stage']]['topics'].append(topic_dict)
        syllabus['all_topics'].append(topic_dict)
        syllabus['topics_by_category']['Core'] = syllabus['topics_by_category'].get('Core', 0) + 1
        
        print(f"  Added: {pt['stage']} - {pt['topic']}")
    
    # Resort by priority
    for stage in syllabus['stages']:
        syllabus['stages'][stage]['topics'].sort(key=lambda x: x['priority'])
        syllabus['stages'][stage]['topic_count'] = len(syllabus['stages'][stage]['topics'])
    
    syllabus['all_topics'].sort(key=lambda x: (x['stage'], x['priority']))
    
    # Update metadata
    syllabus['metadata']['total_topics'] = len(syllabus['all_topics'])
    syllabus['metadata']['generated'] = datetime.now().isoformat()
    syllabus['metadata']['version'] = "3.2"
    syllabus['metadata']['continuous_learning_enabled'] = True
    
    # Update summary
    syllabus['summary']['total_topics'] = len(syllabus['all_topics'])
    syllabus['summary']['by_category'] = syllabus['topics_by_category']
    
    # Rebuild table
    table_rows = []
    for stage in ['Baby', 'Toddler', 'Child', 'Teen', 'Adult']:
        stage_topics = syllabus['stages'][stage]['topics']
        counts = {'Core': 0, 'Artistic': 0, 'Wealth': 0, 'Reverse': 0, 'Accelerator': 0}
        for t in stage_topics:
            counts[t['category']] += 1
        
        table_rows.append([
            stage,
            counts['Core'],
            counts['Artistic'],
            counts['Wealth'],
            counts['Reverse'],
            counts['Accelerator'],
            len(stage_topics)
        ])
    
    total_counts = syllabus['topics_by_category']
    table_rows.append([
        "Total",
        total_counts['Core'],
        total_counts['Artistic'],
        total_counts['Wealth'],
        total_counts['Reverse'],
        total_counts['Accelerator'],
        len(syllabus['all_topics'])
    ])
    
    syllabus['summary']['table']['rows'] = table_rows
    
    with open(syllabus_path, 'w') as f:
        json.dump(syllabus, f, indent=2)
    
    print(f"\nAfter: {len(syllabus['all_topics'])} topics, Core: {syllabus['topics_by_category'].get('Core', 0)}")
    print("\n✅ Persona topics added with continuous_learning flag")

if __name__ == "__main__":
    main()
