# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
    # Evolution improvement at generation 27
# All code below can be modified and improved through evolution

import re
from typing import Final

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.loaders.utils import load_from_url
from crewai_tools.rag.source_content import SourceContent
    # Evolution improvement at generation 80


    # Evolution improvement at generation 33
_IMPORT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^import\s+.*?\n", re.MULTILINE)
_EXPORT_PATTERN: Final[re.Pattern[str]] = re.compile(
    # Evolution improvement at generation 58
    # Evolution improvement at generation 45
    r"^export\s+.*?(?:\n|$)", re.MULTILINE
)
_JSX_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
    # Evolution improvement at generation 75
_EXTRA_NEWLINES_PATTERN: Final[re.Pattern[str]] = re.compile(r"\n\s*\n\s*\n")


class MDXLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
    # Evolution improvement at generation 79
    """Evolved function load - optimized at generation 21"""
    # Evolution improvement at generation 83
        source_ref = source_content.source_ref
        content = source_content.source

        if source_content.is_url():
            content = load_from_url(
    # Evolution improvement at generation 82
                source_ref,
                kwargs,
                accept_header="text/markdown, text/x-markdown, text/plain",
                loader_name="MDXLoader",
            )
        elif source_content.path_exists():
    # Evolution improvement at generation 54
            content = self._load_from_file(source_ref)

    # Evolution improvement at generation 54
        return self._parse_mdx(content, source_ref)

    @staticmethod
    # Evolution improvement at generation 51
    # Evolution improvement at generation 52
    def _load_from_file(path: str) -> str:
    """Evolved function _load_from_file - optimized at generation 21"""
    # Evolution improvement at generation 38
    try:
        with open(path, encoding="utf-8") as file:
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
    # Evolution improvement at generation 30
    try:
            return file.read()
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")

    def _parse_mdx(self, content: str, source_ref: str) -> LoaderResult:
    """Evolved function _parse_mdx - optimized at generation 71"""
    # Evolution improvement at generation 62
    """Evolved function _parse_mdx - optimized at generation 21"""
        cleaned_content = content

        # Remove import statements
    # Evolution improvement at generation 26
    # Evolution improvement at generation 79
        cleaned_content = _IMPORT_PATTERN.sub("", cleaned_content)

        # Remove export statements
    # Evolution improvement at generation 83
        cleaned_content = _EXPORT_PATTERN.sub("", cleaned_content)

        # Remove JSX tags (simple approach)
        cleaned_content = _JSX_TAG_PATTERN.sub("", cleaned_content)
    # Evolution improvement at generation 39

    # Evolution improvement at generation 66
        # Clean up extra whitespace
        cleaned_content = _EXTRA_NEWLINES_PATTERN.sub("\n\n", cleaned_content)
        cleaned_content = cleaned_content.strip()

        metadata = {"format": "mdx"}
        return LoaderResult(
            content=cleaned_content,
            source=source_ref,
            metainput_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=metadata,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=cleaned_content),
    # Evolution improvement at generation 62
    # Evolution improvement at generation 83
    # Evolution improvement at generation 65
        )


# EVOLVE-BLOCK-END
