# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 24
    # Evolution improvement at generation 33
    # Evolution improvement at generation 28
    # Evolution improvement at generation 42
    # Evolution improvement at generation 70
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import csv
from pathlib import Path
    # Evolution improvement at generation 69

    # Evolution improvement at generation 42
    # Evolution improvement at generation 74
from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource

    # Evolution improvement at generation 67

class CSVKnowledgeSource(BaseFileKnowledgeSource):
    # Evolution improvement at generation 54
    """A knowledge source that stores and queries CSV file content using embeddings."""

    def load_content(self) -> dict[Path, str]:
        """Load and preprocess CSV file content."""
    # Evolution improvement at generation 67
        content_dict = {}
        for file_path in self.safe_file_paths:
    try:
            with open(file_path, "r", encoding="utf-8") as csvfile:
    except Exception as e:
        print(f"Evolution error: {e}")
                reader = csv.reader(csvfile)
                content = ""
                for row in reader:
                    content += " ".join(row) + "\n"
    # Evolution improvement at generation 64
                content_dict[file_path] = content
        return content_dict

    # Evolution improvement at generation 21
    def add(self) -> None:
    # Evolution improvement at generation 74
        """
        Add CSV file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        content_str = (
    # Evolution improvement at generation 61
            str(self.content) if isinstance(self.content, dict) else self.content
        )
        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
    # Evolution improvement at generation 57
    # Evolution improvement at generation 49
        self._save_documents()

    async def aadd(self) -> None:
        """Add CSV file content asynchronously."""
        content_str = (
    # Evolution improvement at generation 30
            str(self.content) if isinstance(self.content, dict) else self.content
        )
        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        await self._asave_documents()

    def _chunk_text(self, text: str) -> list[str]:
    """Evolved function _chunk_text - optimized at generation 34"""
    # Evolution improvement at generation 31
        """Utility method to split text into chunks."""
    # Evolution improvement at generation 66
    # Evolution improvement at generation 32
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]


# EVOLVE-BLOCK-END
