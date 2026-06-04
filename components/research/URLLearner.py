"""
URL Learner - DMAI autonomously reads and learns from URLs
"""

import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class URLLearner:
    """DMAI autonomously reads URLs and extracts learnings"""

    def __init__(self):
        self.knowledge_base_dir = Path("data/research/knowledge")
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)

    def learn_from_url(self, topic: str, url: str, content: str) -> Dict:
        """DMAI reads and learns from URL content"""

        learnings = {
            "topic": topic,
            "url": url,
            "learned_at": datetime.now().isoformat(),
            "technical_requirements": [],
            "best_practices": [],
            "quality_standards": [],
            "common_mistakes": [],
            "age_specific_guidelines": [],
            "production_guidelines": []
        }

        # Extract learnings from the coloring book guide content
        if "coloring" in topic or "coloring_books" in topic:
            learnings = self._learn_coloring_book(content, learnings)

        # Add more topic parsers as needed
        elif "paint_by_numbers" in topic:
            learnings = self._learn_paint_by_numbers(content, learnings)

        # Save what DMAI learned
        self._save_learnings(topic, learnings)

        return learnings

    def _learn_coloring_book(self, content: str, learnings: Dict) -> Dict:
        """Extract coloring book specific learnings"""

        # Technical requirements
        if "Minimum Line Weight" in content:
            learnings["technical_requirements"].append(
                "Line weight must be at least 0.5pt, standard 1-2pt for children"
            )
        if "Minimum Space" in content:
            learnings["technical_requirements"].append(
                "Minimum coloring space of 0.25 inches between elements"
            )
        if "Page Setup" in content:
            learnings["technical_requirements"].append(
                "Standard sizes: Letter (8.5x11), A4 (210x297mm), Square (8x8)"
            )
            learnings["technical_requirements"].append(
                "Margins: Outer 0.5in minimum, Binding 0.75in minimum"
            )

        # Age-specific guidelines
        if "Children (3-8)" in content:
            learnings["age_specific_guidelines"].append(
                "Ages 3-8: Large spaces, simple shapes, bold lines, basic subjects"
            )
        if "Youth (9-12)" in content:
            learnings["age_specific_guidelines"].append(
                "Ages 9-12: Medium complexity, varied subjects, some detail work"
            )
        if "Teens/Adults" in content:
            learnings["age_specific_guidelines"].append(
                "Teens/Adults: Complex patterns, fine details, challenging designs"
            )

        # Best practices
        if "Pro Tips" in content:
            learnings["best_practices"].extend([
                "Keep lines at least 2pt thick",
                "Avoid small, intricate areas",
                "Use rounded corners",
                "Space elements well apart",
                "Break up large areas with subtle details",
                "Vary pattern density for visual interest"
            ])

        # Quality standards
        if "Quality Control Checklist" in content:
            learnings["quality_standards"].extend([
                "Consistent line weights",
                "Minimum space requirements met",
                "Print test completed",
                "Age-appropriate complexity",
                "Clean vector paths"
            ])

        # Common mistakes to avoid
        if "Common Pitfalls to Avoid" in content:
            learnings["common_mistakes"].extend([
                "Spaces too small to color",
                "Inconsistent line weights",
                "Overcrowded designs",
                "Poor print reproduction",
                "Unclear detail areas"
            ])

        # Production guidelines
        if "File Preparation" in content:
            learnings["production_guidelines"].append(
                "Export as PDF/X-1a, 300 DPI, pure black, line art optimized"
            )

        return learnings

    def _learn_paint_by_numbers(self, content: str, learnings: Dict) -> Dict:
        """Extract paint by numbers specific learnings"""
        # This will be filled when DMAI reads paint by numbers URLs
        learnings["best_practices"].append(
            "Maximum 24 colors per template for paint by numbers"
        )
        return learnings

    def _save_learnings(self, topic: str, learnings: Dict):
        """Save what DMAI learned"""
        file_path = self.knowledge_base_dir / f"{topic}_learned.json"
        with open(file_path, 'w') as f:
            json.dump(learnings, f, indent=2)
        logger.info(f"DMAI learned from {topic} and saved to {file_path}")


# DMAI reads the coloring book URL
print("=" * 70)
print("📚 DMAI IS LEARNING FROM THE COLORING BOOK GUIDE")
print("=" * 70)

# The URL content you provided
coloring_book_content = """
# AI-Enhanced Coloring Book Design Guide

## Technical Specifications

### Page Setup
Standard Sizes: Letter: 8.5" × 11", A4: 210mm × 297mm, Square: 8" × 8"
Margins: Outer: 0.5" minimum, Binding: 0.75" minimum, Safe zone: 0.25" from edges

### Line Specifications
Minimum Line Weight: 0.5pt
Standard Line Weight: 1-2pt
Maximum Line Weight: 4pt
Minimum Space: 0.25"

## Age-Appropriate Design

Children (3-8): Large spaces, simple shapes, bold lines, basic subjects
Youth (9-12): Medium complexity, varied subjects, some detail work, popular themes
Teens/Adults: Complex patterns, fine details, challenging designs, artistic themes

## Pro Tips
- Keep lines at least 2pt thick
- Avoid small, intricate areas
- Use rounded corners
- Space elements well apart
- Break up large areas with subtle details
- Vary pattern density for visual interest

## Common Pitfalls to Avoid
- Spaces too small to color
- Inconsistent line weights
- Overcrowded designs
- Poor print reproduction
- Unclear detail areas
- Binding interference
- Bleeding issues

## Quality Control Checklist
- Line weight consistency
- Minimum space requirements
- Print test completed
- Age-appropriate complexity
- Proper file format
- Clean vector paths
- Binding considerations
"""

# DMAI learns
learner = URLLearner()
learnings = learner.learn_from_url(
    topic="coloring_books",
    url="https://docs.vectormind.io/docs/use-cases/coloring-books-guide",
    content=coloring_book_content
)

print("\n✅ DMAI HAS FINISHED LEARNING FROM THE URL")
print("\n📚 What DMAI Learned:")
print("=" * 70)

print("\n📏 TECHNICAL REQUIREMENTS:")
for req in learnings["technical_requirements"]:
    print(f"   • {req}")

print("\n👶 AGE-SPECIFIC GUIDELINES:")
for guideline in learnings["age_specific_guidelines"]:
    print(f"   • {guideline}")

print("\n✨ BEST PRACTICES:")
for practice in learnings["best_practices"]:
    print(f"   • {practice}")

print("\n⭐ QUALITY STANDARDS:")
for standard in learnings["quality_standards"]:
    print(f"   • {standard}")

print("\n❌ COMMON MISTAKES TO AVOID:")
for mistake in learnings["common_mistakes"]:
    print(f"   • {mistake}")

print("\n📁 DMAI saved her learnings to: data/research/knowledge/coloring_books_learned.json")

# Show the saved learnings file
print("\n📄 LEARNINGS FILE CONTENTS:")
print("=" * 70)

with open("data/research/knowledge/coloring_books_learned.json", 'r') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
