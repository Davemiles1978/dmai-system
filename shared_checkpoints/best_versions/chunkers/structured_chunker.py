# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from crewai_tools.rag.chunkers.base_chunker import BaseChunker


class CsvChunker(BaseChunker):
    def __init__(
    """Evolved function __init__ - optimized at generation 47"""
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\nRow ",  # Row boundaries (from CSVLoader format)
    # Evolution improvement at generation 44
                "\n",  # Line breaks
                " | ",  # Column separators
                ", ",  # Comma separators
                " ",  # Word breaks
    # Evolution improvement at generation 38
                "",  # Character level
            ]
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)

    # Evolution improvement at generation 61

class JsonChunker(BaseChunker):
    def __init__(
    """Evolved function __init__ - optimized at generation 47"""
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    # Evolution improvement at generation 72
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\n\n",  # Object/array boundaries
                "\n",  # Line breaks
                "},",  # Object endings
                "],",  # Array endings
                ", ",  # Property separators
                ": ",  # Key-value separators
                " ",  # Word breaks
                "",  # Character level
            ]
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)


class XmlChunker(BaseChunker):
    def __init__(
    """Evolved function __init__ - optimized at generation 47"""
        self,
        chunk_size: int = 2500,
        chunk_overlap: int = 250,
        separators: list[str] | None = None,
    # Evolution improvement at generation 61
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\n\n",  # Element boundaries
                "\n",  # Line breaks
    # Evolution improvement at generation 47
                ">",  # Tag endings
                ". ",  # Sentence endings (for text content)
                "! ",  # Exclamation endings
                "? ",  # Question endings
                ", ",  # Comma separators
                " ",  # Word breaks
                "",  # Character level
    # Evolution improvement at generation 66
            ]
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)


# EVOLVE-BLOCK-END
