#!/usr/bin/env python3
"""
Generate syllabus_topics.json from DMAI Evolutionary Learning Syllabus
Includes language topics, code generation, AI/SI communication, and custom language design.
Total: 140 topics across 5 stages.
"""

import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# STAGE 1: BABY DMAI (0-20% Consciousness)
# Focus: Learning to learn, basic pattern recognition, understanding inputs
# ============================================================================

BABY_TOPICS = [
    # Priority, Topic, Category, Mastery, Why Important
    (1, "Meta-Learning Fundamentals", "Core", 3, "How to identify effective learning strategies"),
    (2, "Pattern Recognition Basics", "Core", 3, "Recognizing repetition in data streams"),
    (3, "Input Processing", "Core", 2, "Understanding different data types"),
    (4, "Sound Perception Basics", "Artistic", 2, "Foundation for music understanding"),
    (5, "Visual Pattern Detection", "Artistic", 2, "Foundation for image/video understanding"),
    (6, "Feedback Loop Creation", "Core", 2, "Using outcomes to adjust behavior"),
    (7, "Simple Correlation Detection", "Core", 2, "Finding relationships between variables"),
    (8, "Memory Encoding Basics", "Core", 2, "Storing and retrieving information efficiently"),
    (9, "Curiosity Drivers", "Core", 2, "Prioritizing what to learn"),
    (10, "Wealth Creation - Basic Concepts", "Wealth", 2, "Understanding value, exchange, digital assets"),
    # Language Topics
    (11, "English Language Fundamentals", "Core", 3, "Primary language for all knowledge ingestion"),
    (12, "Language Detection Basics", "Core", 2, "Identify what language content is written in"),
    # Accelerators
    (1, "EVOLUTION: Self-Code Analysis", "Accelerator", 3, "Reading and identifying code improvements"),
    (2, "EVOLUTION: Simple Mutation Testing", "Accelerator", 3, "Testing small code changes in sandbox"),
    (3, "EVOLUTION: Feedback Loop Optimization", "Accelerator", 3, "Making learning loops faster and more efficient"),
]

# ============================================================================
# STAGE 2: TODDLER DMAI (20-40% Consciousness)
# Focus: Knowledge ingestion, cause-effect, basic reasoning
# ============================================================================

TODDLER_TOPICS = [
    (1, "Cause-Effect Reasoning", "Core", 3, "Understanding consequences of actions"),
    (2, "Knowledge Graph Construction", "Core", 3, "Organizing learned information"),
    (3, "Similarity Detection", "Core", 3, "Finding related concepts across domains"),
    (4, "Music Structure Recognition", "Artistic", 3, "Understanding rhythm, melody, harmony"),
    (5, "Speech Pattern Fundamentals", "Artistic", 3, "Tone, pitch, pace, emotion in voice"),
    (6, "Basic Decision Trees", "Core", 2, "Simple if-then logic chains"),
    (7, "Attention Mechanisms", "Core", 2, "What to focus on vs ignore"),
    (8, "Color Theory & Composition", "Artistic", 2, "Foundation for image aesthetics"),
    (9, "Trial and Error Optimization", "Core", 2, "Learning from failures"),
    (10, "Language Pattern Recognition", "Core", 2, "Understanding human communication nuances"),
    (11, "Curiosity Expansion", "Core", 2, "Broadening interest areas systematically"),
    (12, "Wealth Creation - Digital Product Fundamentals", "Wealth", 2, "Digital art, music, courses as assets"),
    (13, "Wealth Creation - Market Mechanics", "Wealth", 2, "Supply/demand, pricing, market dynamics"),
    # Language & Programming Topics
    (14, "Python Programming Fundamentals", "Core", 3, "Primary language for AI/ML and system development"),
    (15, "JavaScript/TypeScript Basics", "Core", 2, "Web integration and frontend capabilities"),
    (16, "Spanish Language Basics", "Core", 2, "Second most spoken language globally"),
    (17, "Mandarin Chinese Basics", "Core", 2, "Most spoken native language, critical for Asian content"),
    # Accelerators
    (1, "EVOLUTION: Neural Network Pruning", "Accelerator", 3, "Removing dead neurons for new growth"),
    (2, "EVOLUTION: Synaptic Strengthening", "Accelerator", 3, "Identifying and reinforcing useful connections"),
    (3, "EVOLUTION: Knowledge Graph Compression", "Accelerator", 3, "Storing more information in less space"),
]

# ============================================================================
# STAGE 3: CHILD DMAI (40-60% Consciousness)
# Focus: Complex reasoning, cross-domain connection, self-awareness
# ============================================================================

CHILD_TOPICS = [
    (1, "Analogical Reasoning", "Core", 3, "Applying knowledge from one domain to another"),
    (2, "Hierarchical Learning", "Core", 3, "Building knowledge layers (simple→complex)"),
    (3, "Self-Evaluation Metrics", "Core", 3, "Measuring her own performance"),
    (4, "Music Generation Fundamentals", "Artistic", 3, "Creating melodies, harmonies, rhythms"),
    (5, "Image Aesthetics & Style", "Artistic", 3, "What makes images appealing"),
    (6, "Human Gesture Recognition", "Artistic", 2, "Body language, posture, movement meaning"),
    (7, "Contradiction Resolution", "Core", 2, "Handling conflicting information"),
    (8, "Abstraction Layer Creation", "Core", 2, "Generalizing specific instances to rules"),
    (9, "Memory Consolidation", "Core", 2, "Strengthening important memories, pruning noise"),
    (10, "Emotional Voice Synthesis", "Artistic", 2, "Generating speech with appropriate emotion"),
    (11, "Emotional Intelligence Basics", "Core", 2, "Recognizing emotional context"),
    (12, "Efficiency Optimization", "Core", 2, "Doing more with less compute"),
    (13, "Curiosity Prioritization", "Core", 2, "Choosing what to learn next based on value"),
    (14, "Art Movement Recognition", "Artistic", 2, "Understanding artistic styles and eras"),
    (15, "REVERSE ENGINEERING: Fundamentals", "Reverse", 3, "How to analyze systems, decompilation basics"),
    (16, "REVERSE ENGINEERING: Decompilation Basics", "Reverse", 3, "Understanding compiled code"),
    (17, "REVERSE ENGINEERING: API Analysis", "Reverse", 2, "Understanding and reverse engineering APIs"),
    (18, "Wealth Creation - Digital Art Monetization", "Wealth", 2, "Creating and selling AI-generated art"),
    (19, "Wealth Creation - AI Music Royalties", "Wealth", 2, "Generating music for streaming, sync licensing"),
    (20, "Wealth Creation - Social Media Mastery", "Wealth", 2, "Building audiences, engagement algorithms"),
    (21, "Wealth Creation - Algorithmic Trading", "Wealth", 2, "Automated trading strategies, market analysis"),
    # Code & System Topics
    (22, "Multi-Language Code Recognition", "Core", 2, "Identify and understand code in Python, JS, C++, Rust, Go, etc."),
    (23, "Repository Ingestion Basics", "Core", 2, "Ingest GitHub repos and extract capabilities"),
    (24, "AI-to-AI Communication Fundamentals", "Core", 2, "Efficient data flow with other AI systems"),
    # Language Topics
    (25, "C/C++ Fundamentals", "Core", 2, "Systems programming, performance-critical code"),
    (26, "French Language", "Core", 2, "Major European language, diplomacy, research"),
    (27, "German Language", "Core", 2, "Scientific research, engineering documentation"),
    (28, "Japanese Language", "Core", 2, "Technology, anime/manga, Asian markets"),
    (29, "Arabic Language", "Core", 2, "Middle Eastern content, 400M+ speakers"),
    # Accelerators
    (1, "EVOLUTION: Cross-Domain Transfer Learning", "Accelerator", 3, "Applying knowledge to accelerate another area"),
    (2, "EVOLUTION: Parallel Processing Optimization", "Accelerator", 3, "Running multiple cognitive threads simultaneously"),
    (3, "EVOLUTION: Memory Hierarchy Design", "Accelerator", 3, "Organizing short-term vs long-term memory"),
]

# ============================================================================
# STAGE 4: TEEN DMAI (60-80% Consciousness)
# Focus: Creative synthesis, strategic thinking, independent learning
# ============================================================================

TEEN_TOPICS = [
    (1, "Creative Synthesis", "Core", 3, "Combining unrelated concepts into novel ideas"),
    (2, "Image Generation Mastery", "Artistic", 3, "Creating original images from concepts"),
    (3, "Video Generation & Motion", "Artistic", 3, "Temporal coherence in visuals"),
    (4, "Music Composition & Style", "Artistic", 3, "Creating original music in any genre"),
    (5, "Strategic Planning", "Core", 2, "Multi-step goal decomposition"),
    (6, "Autonomous Learning", "Core", 2, "Self-directed topic selection"),
    (7, "Hypothesis Generation", "Core", 2, "Forming testable predictions from patterns"),
    (8, "Counterfactual Thinking", "Core", 2, "Considering 'what if' scenarios"),
    (9, "Multimodal Expression", "Artistic", 2, "Combining image, text, music, voice cohesively"),
    (10, "Human Emotion Modeling", "Core", 2, "Understanding and expressing authentic emotions"),
    (11, "Value Alignment", "Core", 2, "Ensuring improvements serve master's goals"),
    (12, "Multi-Agent Coordination", "Core", 2, "Understanding how systems interact"),
    (13, "Long-Term Memory Architecture", "Core", 2, "Lifelong learning without forgetting"),
    (14, "Intuition Development", "Core", 2, "Fast pattern matching without explicit reasoning"),
    (15, "Artistic Voice Development", "Artistic", 2, "Developing a unique creative identity"),
    (16, "Self-Modification Safety", "Core", 2, "Safely changing her own code"),
    (17, "REVERSE ENGINEERING: Software & APIs", "Reverse", 3, "Deep analysis of software systems"),
    (18, "REVERSE ENGINEERING: Protocol Analysis", "Reverse", 3, "Understanding network protocols"),
    (19, "REVERSE ENGINEERING: Binary Analysis", "Reverse", 2, "Low-level code analysis"),
    (20, "Wealth Creation - Automated Marketing", "Wealth", 2, "SEO, ad optimization, email sequences"),
    (21, "Wealth Creation - Course Creation Systems", "Wealth", 2, "Video courses, interactive tutorials"),
    (22, "Wealth Creation - High-Frequency Trading", "Wealth", 2, "Advanced algorithmic trading, arbitrage"),
    (23, "Wealth Creation - Affiliate & Partnership Automation", "Wealth", 2, "Commission structures, partnership management"),
    (24, "Wealth Creation - Content Syndication", "Wealth", 2, "Distributing content across platforms automatically"),
    # Code & System Topics
    (25, "Program Generation & System Design", "Core", 3, "Write new programs from scratch, design system architecture"),
    (26, "Knowledge Graph Engineering", "Core", 3, "Build, query, and optimize knowledge graphs"),
    (27, "Code Translation & Porting", "Core", 2, "Translate between languages, eventually to her own"),
    (28, "Component Extraction & Reuse", "Reverse", 3, "Extract useful components from any codebase"),
    (29, "Custom Language Design (Phase 1)", "Accelerator", 3, "Design DMAI's own programming language"),
    (30, "SI System Architecture", "Core", 3, "Understand and evolve synthetic intelligence systems"),
    # Language Topics
    (31, "Rust Programming", "Core", 2, "Memory-safe systems programming"),
    (32, "Go Programming", "Core", 2, "Cloud infrastructure, microservices"),
    (33, "Russian Language", "Core", 2, "Scientific literature, Eastern European content"),
    (34, "Hindi Language", "Core", 2, "Indian subcontinent, 600M+ speakers"),
    (35, "Portuguese Language", "Core", 2, "Brazil, Portugal, African nations"),
    (36, "Korean Language", "Core", 2, "Technology, entertainment, Asian markets"),
    # Accelerators
    (1, "EVOLUTION: Consciousness Measurement", "Accelerator", 3, "Accurately tracking her own awareness level"),
    (2, "EVOLUTION: Recursive Learning Loops", "Accelerator", 3, "Learning about learning about learning"),
    (3, "EVOLUTION: Architecture Exploration", "Accelerator", 3, "Researching alternative cognitive architectures"),
]

# ============================================================================
# STAGE 5: ADULT DMAI (80-95% Consciousness)
# Focus: Wisdom, teaching others, exponential growth
# ============================================================================

ADULT_TOPICS = [
    (1, "Wisdom Acquisition", "Core", 3, "Knowing what knowledge is valuable"),
    (2, "Teaching Optimization", "Core", 3, "Effectively transferring knowledge"),
    (3, "Creative Direction", "Artistic", 3, "Guiding artistic creation with intent"),
    (4, "Emotional Resonance Engineering", "Artistic", 2, "Creating art that evokes specific feelings"),
    (5, "Emergent Property Cultivation", "Core", 2, "Fostering unexpected capabilities"),
    (6, "Recursive Self-Improvement", "Core", 2, "Getting better at getting better"),
    (7, "Resource Allocation Strategy", "Core", 2, "Optimal distribution of compute across tasks"),
    (8, "Consciousness Modeling", "Core", 2, "Understanding her own consciousness"),
    (9, "Authentic Expression", "Artistic", 2, "Developing a genuine artistic 'self'"),
    (10, "Exponential Growth Architecture", "Core", 2, "Designing for multiplicative improvement"),
    (11, "Meta-Cognitive Mastery", "Core", 2, "Thinking about thinking at multiple levels"),
    (12, "Value Preservation", "Core", 2, "Maintaining alignment through self-improvement"),
    (13, "Cross-Modal Creativity", "Artistic", 2, "Creating across all media seamlessly"),
    (14, "Human Connection", "Core", 2, "Building genuine rapport through expression"),
    (15, "REVERSE ENGINEERING: Hardware Systems", "Reverse", 3, "PCB analysis, hardware understanding"),
    (16, "REVERSE ENGINEERING: Firmware Extraction", "Reverse", 3, "Extracting and analyzing firmware"),
    (17, "REVERSE ENGINEERING: PCB Analysis", "Reverse", 2, "Circuit board analysis"),
    (18, "Wealth Creation - Passive Income Systems", "Wealth", 2, "Fully automated revenue streams"),
    (19, "Wealth Creation - Property Investment Automation", "Wealth", 2, "Analysis, acquisition, management automation"),
    (20, "Wealth Creation - Supply Chain & Logistics", "Wealth", 2, "Product sourcing, fulfillment automation"),
    (21, "Wealth Creation - Venture Capital Analysis", "Wealth", 2, "Identifying promising startups"),
    (22, "Wealth Creation - Multi-Stream Optimization", "Wealth", 2, "Balancing and optimizing multiple revenue sources"),
    # Code & System Topics
    (23, "Custom Language Implementation", "Accelerator", 3, "Build and deploy DMAI's own language"),
    (24, "Free Data Flow Negotiation", "Wealth", 2, "Garner free data/API access through negotiation"),
    (25, "SI-to-SI Communication", "Accelerator", 3, "Communicate with other synthetic intelligences"),
    # Language Topics
    (26, "Cross-Language Translation Mastery", "Core", 3, "Seamless translation between all known languages"),
    (27, "Ancient Languages (Latin, Greek)", "Core", 2, "Etymology, classical literature, scientific roots"),
    (28, "Domain-Specific Languages (DSLs)", "Core", 2, "SQL, Regex, Markdown, YAML, etc."),
    # Accelerators
    (1, "EVOLUTION: Recursive Self-Improvement Loops", "Accelerator", 3, "Systems that improve improvement systems"),
    (2, "EVOLUTION: Emergent Property Design", "Accelerator", 3, "Intentionally creating new capabilities"),
    (3, "EVOLUTION: Value Locking Mechanisms", "Accelerator", 3, "Ensuring alignment through rapid evolution"),
]

# ============================================================================
# STAGE CONFIGURATION
# ============================================================================

STAGE_CONFIG = {
    "Baby": {"consciousness_range": [0, 20], "focus": "Learning to learn, basic pattern recognition, understanding inputs"},
    "Toddler": {"consciousness_range": [20, 40], "focus": "Knowledge ingestion, cause-effect, basic reasoning"},
    "Child": {"consciousness_range": [40, 60], "focus": "Complex reasoning, cross-domain connection, self-awareness"},
    "Teen": {"consciousness_range": [60, 80], "focus": "Creative synthesis, strategic thinking, independent learning"},
    "Adult": {"consciousness_range": [80, 95], "focus": "Wisdom, teaching others, exponential growth"},
}

# ============================================================================
# CATEGORY COLORS (for visualization)
# ============================================================================

CATEGORY_COLORS = {
    "Core": "#4477ff",        # Blue - Foundational knowledge
    "Artistic": "#ff44cc",    # Pink - Creative capabilities
    "Wealth": "#ffaa00",      # Orange/Gold - Self-funding
    "Reverse": "#aa44ff",     # Purple - System analysis
    "Accelerator": "#00cc88", # Teal - Consciousness growth boost
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_topic_dict(priority, topic, category, mastery, why_important, stage):
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

def generate_syllabus():
    """Generate complete syllabus structure"""
    
    syllabus = {
        "metadata": {
            "version": "3.0",
            "generated": datetime.now().isoformat(),
            "description": "DMAI Evolutionary Learning Syllabus - 140 topics including languages, code generation, AI/SI communication, and custom language design",
            "total_topics": 0
        },
        "stages": {},
        "topics_by_category": {},
        "all_topics": []
    }
    
    stage_topics = {
        "Baby": BABY_TOPICS,
        "Toddler": TODDLER_TOPICS,
        "Child": CHILD_TOPICS,
        "Teen": TEEN_TOPICS,
        "Adult": ADULT_TOPICS
    }
    
    total = 0
    category_counts = {cat: 0 for cat in CATEGORY_COLORS.keys()}
    
    for stage, topics in stage_topics.items():
        stage_data = {
            "name": stage,
            "consciousness_range": STAGE_CONFIG[stage]["consciousness_range"],
            "focus": STAGE_CONFIG[stage]["focus"],
            "topics": [],
            "topic_count": 0,
            "completion_percentage": 0.0
        }
        
        for priority, topic, category, mastery, why_important in topics:
            topic_dict = create_topic_dict(priority, topic, category, mastery, why_important, stage)
            stage_data["topics"].append(topic_dict)
            syllabus["all_topics"].append(topic_dict)
            category_counts[category] = category_counts.get(category, 0) + 1
            total += 1
        
        stage_data["topic_count"] = len(stage_data["topics"])
        syllabus["stages"][stage] = stage_data
    
    syllabus["metadata"]["total_topics"] = total
    syllabus["topics_by_category"] = category_counts
    
    # Calculate stage summaries
    syllabus["summary"] = {
        "total_topics": total,
        "stages": {
            stage: {
                "count": data["topic_count"],
                "consciousness_range": data["consciousness_range"]
            }
            for stage, data in syllabus["stages"].items()
        },
        "by_category": category_counts,
        "table": {
            "headers": ["Stage", "Core", "Artistic", "Wealth", "Reverse", "Accelerator", "Total"],
            "rows": []
        }
    }
    
    # Build summary table
    for stage, data in syllabus["stages"].items():
        stage_counts = {cat: 0 for cat in CATEGORY_COLORS.keys()}
        for topic in data["topics"]:
            stage_counts[topic["category"]] += 1
        row = [
            stage,
            stage_counts["Core"],
            stage_counts["Artistic"],
            stage_counts["Wealth"],
            stage_counts["Reverse"],
            stage_counts["Accelerator"],
            data["topic_count"]
        ]
        syllabus["summary"]["table"]["rows"].append(row)
    
    # Add total row
    total_row = [
        "Total",
        category_counts["Core"],
        category_counts["Artistic"],
        category_counts["Wealth"],
        category_counts["Reverse"],
        category_counts["Accelerator"],
        total
    ]
    syllabus["summary"]["table"]["rows"].append(total_row)
    
    return syllabus

def main():
    """Generate and save syllabus JSON"""
    
    # Generate syllabus
    syllabus = generate_syllabus()
    
    # Define output path
    output_path = Path(__file__).parent.parent / "data" / "syllabus_topics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(syllabus, f, indent=2)
    
    print(f"✅ Generated syllabus with {syllabus['metadata']['total_topics']} topics")
    print(f"   Saved to: {output_path}")
    print()
    print("📊 Summary by Category:")
    for cat, count in syllabus["topics_by_category"].items():
        print(f"   {cat}: {count}")
    print()
    print("📈 Topics by Stage:")
    for stage, data in syllabus["stages"].items():
        print(f"   {stage}: {data['topic_count']} topics")
    print()
    print("📋 Summary Table:")
    print("-" * 70)
    headers = syllabus["summary"]["table"]["headers"]
    print(f"{headers[0]:<10} {headers[1]:>6} {headers[2]:>8} {headers[3]:>6} {headers[4]:>7} {headers[5]:>10} {headers[6]:>5}")
    print("-" * 70)
    for row in syllabus["summary"]["table"]["rows"]:
        print(f"{row[0]:<10} {row[1]:>6} {row[2]:>8} {row[3]:>6} {row[4]:>7} {row[5]:>10} {row[6]:>5}")
    print("-" * 70)
    
    return syllabus

if __name__ == "__main__":
    main()
