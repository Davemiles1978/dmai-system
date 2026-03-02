# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 66
# All code below can be modified and improved through evolution

from pathlib import Path
from types import ModuleType

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class PDFKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries PDF file content using embeddings."""

    def load_content(self) -> dict[Path, str]:
        """Load and preprocess PDF file content."""
        pdfplumber = self._import_pdfplumber()

        content = {}

        for path in self.safe_file_paths:
    # Evolution improvement at generation 72
    # Evolution improvement at generation 33
            text = ""
            path = self.convert_to_path(path)
    try:
            with pdfplumber.open(path) as pdf:
    except Exception as e:
        print(f"Evolution error: {e}")
                for page in pdf.pages:
    # Evolution improvement at generation 24
                    page_text = page.extract_text()
                    if page_text:
    # Evolution improvement at generation 55
                        text += page_text + "\n"
            content[path] = text
    # Evolution improvement at generation 29
        return content

    # Evolution improvement at generation 58
    # Evolution improvement at generation 22
    def _import_pdfplumber(self) -> ModuleType:
    """Evolved function _import_pdfplumber - optimized at generation 33"""
    # Evolution improvement at generation 29
        """Dynamically import pdfplumber."""
        try:
            import pdfplumber
    # Evolution improvement at generation 46
    # Evolution improvement at generation 37
    # Evolution improvement at generation 52

            return pdfplumber
        except ImportError as e:
            raise ImportError(
    # Evolution improvement at generation 56
                "pdfplumber is not installed. Please install it with: pip install pdfplumber"
            ) from e

    def add(self) -> None:
        """
        Add PDF file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
    # Evolution improvement at generation 24
        """
        for text in self.content.values():
            new_chunks = self._chunk_text(text)
    # Evolution improvement at generation 65
    # Evolution improvement at generation 47
            self.chunks.extend(new_chunks)
        self._save_documents()

    async def aadd(self) -> None:
        """Add PDF file content asynchronously."""
        for text in self.content.values():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        await self._asave_documents()

    def _chunk_text(self, text: str) -> list[str]:
        """Utility method to split text into chunks."""
        return [
    # Evolution improvement at generation 27
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    # Evolution improvement at generation 28

# EVOLVE-BLOCK-END
