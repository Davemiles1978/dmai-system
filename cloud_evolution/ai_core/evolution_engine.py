import os
import json
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """DMAI's self-improvement system"""
    
    def __init__(self, evolution_dir='ai_core/evolution'):
        self.evolution_dir = evolution_dir
        os.makedirs(evolution_dir, exist_ok=True)
        self.generation = self.load_generation()
        self.evolution_history = self.load_history()
        self.best_score = self.load_best_score()
        
    def load_generation(self):
        gen_file = os.path.join(self.evolution_dir, 'current_generation.txt')
        if os.path.exists(gen_file):
            with open(gen_file, 'r') as f:
                return int(f.read().strip())
        return 1
    
    def load_history(self):
        hist_file = os.path.join(self.evolution_dir, 'evolution_history.json')
        if os.path.exists(hist_file):
            with open(hist_file, 'r') as f:
                return json.load(f)
        return []
    
    def load_best_score(self):
        score_file = os.path.join(self.evolution_dir, 'best_score.txt')
        if os.path.exists(score_file):
            with open(score_file, 'r') as f:
                return float(f.read().strip())
        return 0.0
    
    def evolve(self, brain_state, performance_metrics):
        """Run one evolution cycle"""
        logger.info(f"🧬 Starting evolution cycle {self.generation}")
        
        evolution = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "improvements": [],
            "score": 0,
            "success": False
        }
        
        # Analyze current performance
        weak_areas = self.identify_weaknesses(brain_state, performance_metrics)
        
        # Generate improvements for weak areas
        for area in weak_areas:
            improvement = self.generate_improvement(area, brain_state)
            if improvement:
                evolution["improvements"].append(improvement)
        
        # If we have improvements, test them
        if evolution["improvements"]:
            test_result = self.test_improvements(evolution["improvements"], brain_state)
            evolution["score"] = test_result["score"]
            
            # If improvement is better than current best, implement it
            if evolution["score"] > self.best_score:
                self.implement_improvements(evolution["improvements"])
                self.best_score = evolution["score"]
                evolution["success"] = True
                logger.info(f"✅ Evolution successful! New score: {self.best_score}")
            else:
                logger.info(f"📊 Evolution score {evolution['score']} didn't beat best {self.best_score}")
        
        # Record evolution
        evolution["best_score"] = self.best_score
        self.evolution_history.append(evolution)
        self.save_history()
        
        # Increment generation
        self.generation += 1
        self.save_generation()
        self.save_best_score()
        
        return evolution
    
    def identify_weaknesses(self, brain_state, metrics):
        """Find areas that need improvement"""
        weaknesses = []
        
        # Check capability levels
        capabilities = brain_state.get("capabilities", {})
        for cap, data in capabilities.items():
            level = data.get("level", 0)
            if level < 0.3:
                weaknesses.append(f"low_{cap}")
            elif level < 0.6 and random.random() > 0.7:  # Randomly target medium skills
                weaknesses.append(f"improve_{cap}")
        
        # Check performance metrics
        if metrics.get("accuracy", 1) < 0.8:
            weaknesses.append("accuracy")
        if metrics.get("speed", 1) < 0.5:
            weaknesses.append("speed")
        if metrics.get("knowledge_gaps", 0) > 10:
            weaknesses.append("knowledge")
            
        # Limit to top 3 weaknesses
        return weaknesses[:3]
    
    def generate_improvement(self, area, brain_state):
        """Create an improvement for a weak area"""
        
        # Improvement templates based on area
        improvement_templates = {
            "low_reasoning": {
                "name": "Enhanced Reasoning",
                "description": "Implement chain-of-thought reasoning",
                "expected_improvement": 0.2,
                "type": "algorithm"
            },
            "low_memory": {
                "name": "Memory Expansion",
                "description": "Add vector database for long-term memory",
                "expected_improvement": 0.25,
                "type": "architecture"
            },
            "low_learning": {
                "name": "Continuous Learning",
                "description": "Enable online learning from interactions",
                "expected_improvement": 0.3,
                "type": "algorithm"
            },
            "low_creation": {
                "name": "Creation Enhancement",
                "description": "Improve generation quality with better models",
                "expected_improvement": 0.2,
                "type": "model"
            },
            "low_analysis": {
                "name": "Deeper Analysis",
                "description": "Add multi-pass analysis capability",
                "expected_improvement": 0.15,
                "type": "algorithm"
            },
            "accuracy": {
                "name": "Accuracy Boost",
                "description": "Add verification step before responding",
                "expected_improvement": 0.1,
                "type": "validation"
            },
            "speed": {
                "name": "Speed Optimization",
                "description": "Optimize model inference with quantization",
                "expected_improvement": 0.3,
                "type": "optimization"
            },
            "knowledge": {
                "name": "Knowledge Expansion",
                "description": "Connect to external knowledge sources",
                "expected_improvement": 0.2,
                "type": "integration"
            }
        }
        
        # Check if we have a template for this area
        for key, template in improvement_templates.items():
            if key in area:
                return {
                    "area": area,
                    "name": template["name"],
                    "description": template["description"],
                    "expected_improvement": template["expected_improvement"],
                    "type": template["type"],
                    "confidence": random.uniform(0.6, 0.9)
                }
        
        # Generic improvement if no template matches
        return {
            "area": area,
            "name": f"Improve {area}",
            "description": f"Generic improvement for {area}",
            "expected_improvement": 0.1,
            "type": "general",
            "confidence": random.uniform(0.5, 0.7)
        }
    
    def test_improvements(self, improvements, brain_state):
        """Test if improvements actually help"""
        # In real implementation, this would run actual tests
        # For now, simulate testing
        
        base_score = self.best_score if self.best_score > 0 else 0.5
        
        # Calculate potential improvement
        total_expected = sum(i.get("expected_improvement", 0.1) for i in improvements)
        
        # Add randomness (not all improvements work as expected)
        actual_improvement = total_expected * random.uniform(0.5, 1.2)
        
        new_score = min(1.0, base_score + actual_improvement)
        
        return {
            "score": new_score,
            "improvement": actual_improvement,
            "tested": datetime.now().isoformat(),
            "details": [f"Tested {i['name']}" for i in improvements]
        }
    
    def implement_improvements(self, improvements):
        """Apply successful improvements"""
        # Save improvements to file
        impl_file = os.path.join(self.evolution_dir, f'generation_{self.generation}_improvements.json')
        with open(impl_file, 'w') as f:
            json.dump({
                "generation": self.generation,
                "timestamp": datetime.now().isoformat(),
                "improvements": improvements
            }, f, indent=2)
        
        # Log what was implemented
        logger.info(f"📝 Implemented {len(improvements)} improvements")
        for imp in improvements:
            logger.info(f"  - {imp['name']}: {imp['description']}")
    
    def save_history(self):
        hist_file = os.path.join(self.evolution_dir, 'evolution_history.json')
        with open(hist_file, 'w') as f:
            json.dump(self.evolution_history, f, indent=2)
    
    def save_generation(self):
        gen_file = os.path.join(self.evolution_dir, 'current_generation.txt')
        with open(gen_file, 'w') as f:
            f.write(str(self.generation))
    
    def save_best_score(self):
        score_file = os.path.join(self.evolution_dir, 'best_score.txt')
        with open(score_file, 'w') as f:
            f.write(str(self.best_score))
    
    def get_evolution_summary(self):
        """Get summary of evolution progress"""
        return {
            "current_generation": self.generation,
            "best_score": self.best_score,
            "total_evolutions": len(self.evolution_history),
            "last_evolution": self.evolution_history[-1] if self.evolution_history else None
        }

if __name__ == "__main__":
    engine = EvolutionEngine()
    print(f"🧬 Evolution Engine ready")
    print(f"Current generation: {engine.generation}")
    print(f"Best score: {engine.best_score}")
    
    # Test with dummy data
    test_brain = {
        "capabilities": {
            "reasoning": {"level": 0.2},
            "memory": {"level": 0.1},
            "creation": {"level": 0.3}
        }
    }
    test_metrics = {
        "accuracy": 0.7,
        "speed": 0.4,
        "knowledge_gaps": 15
    }
    
    result = engine.evolve(test_brain, test_metrics)
    print(f"\nEvolution result: {result['success']}")
    print(f"New score: {result['score']}")
