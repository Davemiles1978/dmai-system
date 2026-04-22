#!/usr/bin/env python3
"""
Patch syllabus_topics.json to add the 3 missing Artistic topics
"""

import json
from pathlib import Path
from datetime import datetime

# Missing Artistic topics
MISSING_TOPICS = [
    {
        "stage": "Child",
        "priority": 30,  # After Arabic Language (29)
        "topic": "Visual Storytelling Basics",
        "category": "Artistic",
        "mastery": 2,
        "why_important": "Narrative through imagery, composition for emotional impact"
    },
    {
        "stage": "Teen",
        "priority": 37,  # After Korean Language (36)
        "topic": "Interactive Art & Installation",
        "category": "Artistic",
        "mastery": 2,
        "why_important": "Creating responsive, immersive artistic experiences"
    },
    {
        "stage": "Adult",
        "priority": 29,  # After DSLs (28)
        "topic": "Artistic Legacy & Curation",
        "category": "Artistic",
        "mastery": 2,
        "why_important": "Building a cohesive body of work, curating artistic identity over time"
    }
]

CATEGORY_COLORS = {
    "Core": "#4477ff",
    "Artistic": "#ff44cc",
    "Wealth": "#ffaa00",
    "Reverse": "#aa44ff",
    "Accelerator": "#00cc88",
}

def create_topic_dict(stage, priority, topic, category, mastery, why_important):
    """Create a standardized topic dictionary"""
    return {
        "id": f"topic_{stage.lower()}_{priority:02d}",
        "topic": topic,
        "category": category,
        "color": CATEGORY_COLORS.get(category, "#888888"),
        "stage": stage,
        "priority": priority,
        "mastery_required": mastery,
        "mastery_passes": mastery,
        "why_important": why_important,
        "status": "not_started",
        "progress": 0.0,
        "micro_neurons_created": 0,
        "synapses_created": 0,
        "last_updated": None
    }

def main():
    syllabus_path = Path(__file__).parent.parent / "data" / "syllabus_topics.json"
    
    # Load existing
    with open(syllabus_path, 'r') as f:
        syllabus = json.load(f)
    
    print(f"Before patch: {syllabus['metadata']['total_topics']} topics")
    print(f"  Artistic: {syllabus['topics_by_category'].get('Artistic', 0)}")
    
    # Add missing topics
    for missing in MISSING_TOPICS:
        stage = missing['stage']
        topic_dict = create_topic_dict(
            missing['stage'],
            missing['priority'],
            missing['topic'],
            missing['category'],
            missing['mastery'],
            missing['why_important']
        )
        
        # Add to stage
        syllabus['stages'][stage]['topics'].append(topic_dict)
        syllabus['stages'][stage]['topic_count'] += 1
        
        # Add to all_topics
        syllabus['all_topics'].append(topic_dict)
        
        # Update category count
        syllabus['topics_by_category']['Artistic'] = syllabus['topics_by_category'].get('Artistic', 0) + 1
        
        print(f"  Added: {stage} - {missing['topic']}")
    
    # Resort topics by priority within each stage
    for stage in syllabus['stages']:
        syllabus['stages'][stage]['topics'].sort(key=lambda x: x['priority'])
    
    syllabus['all_topics'].sort(key=lambda x: (x['stage'], x['priority']))
    
    # Update metadata
    syllabus['metadata']['total_topics'] = len(syllabus['all_topics'])
    syllabus['metadata']['generated'] = datetime.now().isoformat()
    syllabus['metadata']['version'] = "3.1"
    syllabus['metadata']['description'] = "DMAI Evolutionary Learning Syllabus - 140 topics (patched with missing Artistic topics)"
    
    # Update summary
    syllabus['summary']['total_topics'] = len(syllabus['all_topics'])
    syllabus['summary']['by_category'] = syllabus['topics_by_category']
    
    # Update summary table
    stage_counts = {}
    for stage, data in syllabus['stages'].items():
        counts = {cat: 0 for cat in CATEGORY_COLORS.keys()}
        for topic in data['topics']:
            counts[topic['category']] += 1
        stage_counts[stage] = counts
        syllabus['summary']['stages'][stage]['count'] = data['topic_count']
    
    # Rebuild table
    table_rows = []
    for stage in ['Baby', 'Toddler', 'Child', 'Teen', 'Adult']:
        counts = stage_counts[stage]
        table_rows.append([
            stage,
            counts['Core'],
            counts['Artistic'],
            counts['Wealth'],
            counts['Reverse'],
            counts['Accelerator'],
            syllabus['stages'][stage]['topic_count']
        ])
    
    # Total row
    total_counts = syllabus['topics_by_category']
    table_rows.append([
        "Total",
        total_counts['Core'],
        total_counts['Artistic'],
        total_counts['Wealth'],
        total_counts['Reverse'],
        total_counts['Accelerator'],
        syllabus['metadata']['total_topics']
    ])
    
    syllabus['summary']['table']['rows'] = table_rows
    
    # Save
    with open(syllabus_path, 'w') as f:
        json.dump(syllabus, f, indent=2)
    
    print(f"\nAfter patch: {syllabus['metadata']['total_topics']} topics")
    print(f"  Artistic: {syllabus['topics_by_category'].get('Artistic', 0)}")
    print(f"\n✅ Saved to: {syllabus_path}")

if __name__ == "__main__":
    main()
