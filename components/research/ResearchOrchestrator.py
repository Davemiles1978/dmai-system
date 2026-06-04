"""
DMAI Research Orchestrator - Self-Education System
Learns from approved URLs and documents best practices before generating content
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """DMAI's autonomous research and learning system"""

    def __init__(self, ai_hub=None):
        self.ai_hub = ai_hub
        self.research_topics = {}
        self.learned_best_practices = {}
        self.research_complete = False
        self.knowledge_base_dir = Path("data/research/knowledge")
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)

        # Define research curriculum
        self.curriculum = {
            "coloring_books": {
                "urls": [
                    "https://docs.vectormind.io/docs/use-cases/coloring-books-guide",
                    "https://skywork.ai/blog/doc/ai-coloring-book-generator-ultimate-guide/",
                    "https://www.promotionchoice.com/blog/printing-artwork-for-children-coloring-books-rules-examples-and-common-mistakes.html"
                ],
                "questions": [
                    "What are the technical requirements for coloring book pages?",
                    "What line weight is appropriate for different age groups?",
                    "What makes a good coloring page vs a poor one?",
                    "How do I ensure each page is unique but consistent in style?",
                    "What are the commercial requirements for print-ready coloring books?"
                ]
            },
            "paint_by_numbers": {
                "urls": [
                    "https://tucocoo.com/it/blogs/paint-by-numbers/how-to-turn-cool-art-into-paint-by-numbers-template",
                    "https://www.ravensburger.ie/en-IE/brio/common/service/faq-photo-gifts/faq-painting-by-numbers"
                ],
                "questions": [
                    "How are paint-by-number templates created from source art?",
                    "What is the maximum number of colors for a template?",
                    "How do I simplify complex images into numbered sections?",
                    "What makes a good paint-by-numbers template?"
                ]
            },
            "cartoon_drawings": {
                "urls": [
                    "https://getillustrations.com/single-illustration/reading-book-learning-student-knowledge",
                    "https://getillustrations.com/single-illustration/history-library-history-book-dinosaur-landmarks"
                ],
                "questions": [
                    "What makes a cartoon illustration engaging for children?",
                    "What are the characteristics of good children's book illustrations?",
                    "How do I maintain consistent character design across pages?",
                    "What is the appropriate complexity for different age groups?"
                ]
            },
            "book_writing": {
                "urls": [
                    "https://kdp.amazon.com/en_US/help/topic/G200735180",
                    "https://www.writingclasses.com/toolbox/ask-writer/how-to-write-a-novel"
                ],
                "questions": [
                    "What is the standard structure for a novel?",
                    "How do I develop compelling characters?",
                    "What makes a good opening chapter?",
                    "How do I pace a story effectively?",
                    "What are the genre conventions for fiction, mystery, romance, sci-fi?"
                ]
            },
            "technical_manuals": {
                "urls": [
                    "https://www.writerswrite.com/technical-writing/guidelines-for-writing-technical-manuals/"
                ],
                "questions": [
                    "What is the standard structure for a technical manual?",
                    "How do I write clear instructions for beginners?",
                    "What visual elements should be included?",
                    "How do I organize chapters logically?"
                ]
            },
            "article_writing": {
                "urls": [
                    "https://www.wordstream.com/blog/ws/2020/03/25/how-to-write-a-blog-post"
                ],
                "questions": [
                    "What makes an engaging article headline?",
                    "How do I structure a blog post for readability?",
                    "What is the optimal article length for different platforms?",
                    "How do I incorporate SEO without sacrificing quality?"
                ]
            },
            "research_papers": {
                "urls": [
                    "https://www.nature.com/nature/for-authors/formatting-guide"
                ],
                "questions": [
                    "What is the standard IMRaD structure for research papers?",
                    "How do I write an effective abstract?",
                    "What citation formats are required for academic publishing?",
                    "How do I present data and findings professionally?"
                ]
            },
            "mock_exams": {
                "urls": [
                    "https://www.teachervision.com/test-prep/creating-effective-mock-exams"
                ],
                "questions": [
                    "How do I structure a mock exam?",
                    "What types of questions should be included?",
                    "How do I set appropriate difficulty levels?",
                    "What is the standard format for answer keys?"
                ]
            },
            "how_to_draw_books": {
                "urls": [
                    "https://www.artistsnetwork.com/art-instruction/how-to-write-a-how-to-draw-book/"
                ],
                "questions": [
                    "How do I break down drawing into simple steps?",
                    "What makes an effective how-to-draw lesson?",
                    "How do I structure instructions for beginners?",
                    "What visual progression works best for learning?"
                ]
            },
            "childrens_books": {
                "urls": [
                    "https://www.writingclasses.com/toolbox/ask-writer/how-to-write-a-childrens-book"
                ],
                "questions": [
                    "What are the age-appropriate word counts for children's books?",
                    "How do I structure a picture book vs a chapter book?",
                    "What themes work best for different age groups?",
                    "How do illustrations complement the text?"
                ]
            }
        }

    def research_topic(self, topic: str) -> Dict:
        """Research a specific topic by analyzing its URLs"""
        if topic not in self.curriculum:
            return {"error": f"Topic '{topic}' not in curriculum"}

        topic_data = self.curriculum[topic]
        findings = {
            "topic": topic,
            "researched_at": datetime.now().isoformat(),
            "sources": [],
            "best_practices": [],
            "technical_requirements": [],
            "quality_standards": [],
            "common_mistakes": []
        }

        for url in topic_data["urls"]:
            findings["sources"].append({
                "url": url,
                "status": "pending_review",
                "message": "URL requires manual review for appropriateness before DMAI can learn from it"
            })

        findings["research_questions"] = topic_data["questions"]
        self.research_topics[topic] = findings
        self._save_research(topic, findings)
        return findings

    def add_manual_best_practice(self, topic: str, practice: str, category: str):
        """Manually add best practices from reviewed content"""
        if topic not in self.research_topics:
            self.research_topics[topic] = {"topic": topic, "best_practices": []}

        self.research_topics[topic]["best_practices"].append({
            "practice": practice,
            "category": category,
            "added_at": datetime.now().isoformat()
        })
        self._save_research(topic, self.research_topics[topic])

    def _save_research(self, topic: str, data: Dict):
        """Save research to file"""
        file_path = self.knowledge_base_dir / f"{topic}_research.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_research_summary(self, topic: str) -> Dict:
        """Get summary of research for a topic"""
        file_path = self.knowledge_base_dir / f"{topic}_research.json"
        if file_path.exists():
            with open(file_path) as f:
                return json.load(f)
        return {"error": f"No research found for {topic}"}

    def list_topics(self) -> List[str]:
        """List all research topics"""
        return list(self.curriculum.keys())

    def generate_learning_summary(self) -> str:
        """Generate a summary of what DMAI has learned"""
        summary = "# DMAI Research Learning Summary\n\n"
        summary += f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for topic, data in self.research_topics.items():
            summary += f"## {topic.replace('_', ' ').title()}\n\n"
            if data.get("best_practices"):
                summary += "### Best Practices Learned:\n"
                for bp in data["best_practices"]:
                    summary += f"- {bp['practice']}\n"
            summary += "\n"

        return summary


def create_research_guide():
    """Create a guide for you to provide research feedback to DMAI"""
    guide = """# DMAI Book Creation Research Guide

## Purpose
This guide helps you provide DMAI with the best practices she needs to create professional, publishable books.

## How to Use This
1. Review the URLs provided
2. For each URL, extract key learnings
3. Add best practices using the commands in the Python console

## Current Research Status
The research system is ready. DMAI has initialized research files for all 10 topics.
Once you provide best practices from the URLs, she will apply them to future book generation.
"""
    return guide


if __name__ == "__main__":
    print("=" * 70)
    print("📚 DMAI RESEARCH ORCHESTRATOR")
    print("=" * 70)

    r = ResearchOrchestrator()

    print("\n📋 Research Topics Available:")
    for topic in r.list_topics():
        print(f"   • {topic.replace('_', ' ').title()}")

    print("\n🔍 Please review the URLs for each topic.")
    print("   Once reviewed, provide the key learnings to DMAI.")
    print("\n📁 Research data saved to: data/research/knowledge/")

    # Save the research guide
    guide = create_research_guide()
    guide_path = Path("data/research/RESEARCH_GUIDE.md")
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    with open(guide_path, 'w') as f:
        f.write(guide)

    print(f"\n📘 Research guide saved to: {guide_path}")

    # Initialize research for all topics
    print("\n📖 Initializing research files...")
    for topic in r.list_topics():
        result = r.research_topic(topic)
        print(f"   ✓ {topic}")

    print("\n✅ Research system ready.")
    print("\n⏳ DMAI is waiting for you to provide best practices from the URLs.")
