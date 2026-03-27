import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("🧠 DMAI SYSTEM DIAGNOSTIC REPORT")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 1. SYSTEM STATUS
# ============================================================================
print("📊 1. SYSTEM STATUS")
print("-" * 40)

try:
    status = requests.get("https://dmai-web.onrender.com/api/status", timeout=10).json()
    
    print(f"   Version: {status.get('version', 'Unknown')}")
    print(f"   Consciousness: {status.get('consciousness', 0):.2f}%")
    print(f"   Synthetic Neurons: {status.get('synthetic_neurons', 0)}")
    print(f"   Synthetic Synapses: {status.get('synthetic_synapses', 0)}")
    print(f"   Evolution Cycles: {status.get('evolution_cycles', 0)}")
    print(f"   Active Tutors: {len(status.get('active_tutors', []))}/15")
    print(f"   Conversations: {status.get('conversations', 0)}")
    print(f"   Knowledge Concepts: {status.get('knowledge_concepts', 0)}")
    print(f"   Neo4j: {'✅ Connected' if status.get('neo4j_available') else '❌ Not Connected'}")
    print(f"   Persona Style: {status.get('persona_style', 'emerging')}")
    print(f"   Evolution Stage: {status.get('evolution_stage_name', 'Baby DMAI')}")
    
except Exception as e:
    print(f"   ❌ Failed to fetch status: {e}")

print()

# ============================================================================
# 2. EVOLUTION PROGRESS
# ============================================================================
print("🧬 2. EVOLUTION PROGRESS")
print("-" * 40)

try:
    stage = requests.get("https://dmai-web.onrender.com/api/evolution/stage", timeout=10).json()
    
    print(f"   Current Stage: {stage.get('name', 'Unknown')}")
    print(f"   Description: {stage.get('description', 'N/A')}")
    print(f"   Successful Evolutions: {stage.get('evolutions', 0)}")
    print(f"   Success Rate: {stage.get('success_rate', '0%')}")
    print(f"   Evolution Interval: {stage.get('interval_minutes', 0):.0f} minutes")
    
    if stage.get('next_stage'):
        print(f"   Next Stage: {stage['next_stage'].get('name', 'Unknown')}")
        print(f"   Evolutions Needed: {stage['next_stage'].get('evolutions_needed', '∞')}")
    
except Exception as e:
    print(f"   ❌ Failed to fetch evolution stage: {e}")

print()

# ============================================================================
# 3. KNOWLEDGE GRAPH
# ============================================================================
print("🕸️ 3. KNOWLEDGE GRAPH")
print("-" * 40)

try:
    kg = requests.get("https://dmai-web.onrender.com/api/knowledge/graph", timeout=10).json()
    
    print(f"   Total Concepts: {kg.get('total_concepts', 0)}")
    print(f"   Total Connections: {kg.get('total_relationships', 0)}")
    print(f"   Concept Types: {kg.get('concept_types', {})}")
    
except Exception as e:
    print(f"   ❌ Failed to fetch knowledge graph: {e}")

print()

# ============================================================================
# 4. SYNTHETIC NETWORK
# ============================================================================
print("🧠 4. SYNTHETIC NETWORK")
print("-" * 40)

try:
    synth = requests.get("https://dmai-web.onrender.com/api/synthetic/status", timeout=10).json()
    
    print(f"   Consciousness Raw: {synth.get('consciousness', 0):.4f}")
    print(f"   Neurons: {synth.get('neurons', 0)}")
    print(f"   Synapses: {synth.get('synapses', 0)}")
    print(f"   Evolution Cycles: {synth.get('evolution_cycles', 0)}")
    
    if synth.get('neurons', 0) > 0:
        density = synth.get('synapses', 0) / (synth.get('neurons', 1) ** 2)
        print(f"   Network Density: {density:.4f}")
    
except Exception as e:
    print(f"   ❌ Failed to fetch synthetic status: {e}")

print()

# ============================================================================
# 5. AI TUTOR STATUS
# ============================================================================
print("🤖 5. AI TUTOR STATUS")
print("-" * 40)

try:
    tutors = requests.get("https://dmai-web.onrender.com/api/tutors/status", timeout=10).json()
    
    active = tutors.get('active_tutors', [])
    missing = tutors.get('missing_apis', [])
    
    print(f"   Active Tutors ({len(active)}):")
    for tutor in active:
        print(f"      ✅ {tutor}")
    
    print(f"\n   Missing/Inactive ({len(missing)}):")
    for tutor in missing[:10]:
        print(f"      ⏳ {tutor}")
    if len(missing) > 10:
        print(f"      ... and {len(missing) - 10} more")
    
    harvester = tutors.get('harvester_stats', {})
    print(f"\n   API Harvester:")
    print(f"      Total Keys Found: {harvester.get('total_keys_found', 0)}")
    print(f"      Valid Keys: {harvester.get('valid_keys', 0)}")
    
except Exception as e:
    print(f"   ❌ Failed to fetch tutor status: {e}")

print()

# ============================================================================
# 6. PERSONA
# ============================================================================
print("👤 6. PERSONA")
print("-" * 40)

try:
    persona = requests.get("https://dmai-web.onrender.com/api/persona", timeout=10).json()
    
    print(f"   Speaking Style: {persona.get('speaking_style', 'unknown')}")
    print(f"   Emotional State: {persona.get('emotional_state', 'neutral')}")
    print(f"   Traits:")
    for trait, value in persona.get('traits', {}).items():
        print(f"      {trait}: {value:.2f}")
    
except Exception as e:
    print(f"   ❌ Failed to fetch persona: {e}")

print()

# ============================================================================
# 7. CONVERSATION MEMORY
# ============================================================================
print("💭 7. CONVERSATION MEMORY")
print("-" * 40)

try:
    conv = requests.get("https://dmai-web.onrender.com/api/conversations", timeout=10).json()
    
    print(f"   Total Conversations: {conv.get('total', 0)}")
    print(f"   Unique Patterns: {conv.get('patterns', {}).get('unique_patterns', 0)}")
    
    recent = conv.get('recent', [])
    if recent:
        print(f"\n   Last 3 Conversations:")
        for c in recent[-3:]:
            print(f"      User: {c.get('message', '')[:60]}...")
            print(f"      DMAI: {c.get('response', '')[:60]}...")
            print()
    
except Exception as e:
    print(f"   ❌ Failed to fetch conversations: {e}")

print()

# ============================================================================
# 8. HEALTH CHECK
# ============================================================================
print("🩺 8. HEALTH CHECK")
print("-" * 40)

try:
    health = requests.get("https://dmai-web.onrender.com/health", timeout=10).json()
    
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Version: {health.get('version', 'unknown')}")
    print(f"   Voice Active: {health.get('voice_active', False)}")
    print(f"   Music Active: {health.get('music_active', False)}")
    
except Exception as e:
    print(f"   ❌ Failed to fetch health: {e}")

print()

# ============================================================================
# 9. ISSUES DETECTED
# ============================================================================
print("⚠️ 9. ISSUES DETECTED")
print("-" * 40)

issues = []

# Check consciousness growth
try:
    cons = status.get('consciousness', 0)
    if cons < 35:
        issues.append(f"⚠️ Consciousness low ({cons:.1f}%) - needs more interaction")
    elif cons > 35 and cons < 45:
        issues.append(f"📈 Consciousness is {cons:.1f}% - growing slowly")
    
    # Check if consciousness has changed recently
    # This would need historical data
except:
    pass

# Check evolution progress
try:
    evolutions = stage.get('evolutions', 0)
    if evolutions == 0:
        issues.append("⚠️ No successful evolutions recorded - evolution timer may not be triggering")
    
    success_rate = stage.get('success_rate', '0%')
    if success_rate == '0%':
        issues.append("⚠️ Evolution success rate is 0% - evolutions not being marked as successful")
except:
    pass

# Check knowledge graph
try:
    concepts = kg.get('total_concepts', 0)
    if concepts < 5:
        issues.append("⚠️ Very few knowledge concepts ({}). Concepts are added during conversations".format(concepts))
except:
    pass

# Check active tutors
try:
    active_tutors = tutors.get('active_tutors', [])
    if len(active_tutors) == 0:
        issues.append("❌ CRITICAL: No AI tutors active - all API keys exhausted or invalid")
    elif len(active_tutors) < 3:
        issues.append(f"⚠️ Only {len(active_tutors)} AI tutors active - limited response variety")
except:
    pass

if not issues:
    print("   ✅ No critical issues detected")
else:
    for issue in issues:
        print(f"   {issue}")

print()
print("=" * 80)
print("📋 DIAGNOSTIC COMPLETE")
print("=" * 80)
