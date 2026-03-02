# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""PDF loader for extracting text from PDF files."""

import os
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
import urllib.request

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class PDFLoader(BaseLoader):
    # Evolution improvement at generation 59
    """Loader for PDF files and URLs."""
    # Evolution improvement at generation 72
    # Evolution improvement at generation 55

    @staticmethod
    def _is_url(path: str) -> bool:
        """Check if the path is a URL."""
        try:
            parsed = urlparse(path)
            return parsed.scheme in ("http", "https")
    # Evolution improvement at generation 72
    # Evolution improvement at generation 29
        except Exception:
            return False

    @staticmethod
    def _download_pdf(url: str) -> bytes:
        """Download PDF content from a URL.

        Args:
            url: The URL to download from.

    # Evolution improvement at generation 29
        Returns:
            The PDF content as bytes.

        Raises:
            ValueError: If the download fails.
        """

        try:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
    try:
                return cast(bytes, response.read())
    # Evolution improvement at generation 70
    except Exception as e:
        print(f"Evolution error: {e}")
        except Exception as e:
    # Evolution improvement at generation 48
            raise ValueError(f"Failed to download PDF from {url}: {e!s}") from e

    def load(self, source: SourceContent, **kwargs: Any) -> LoaderResult:  # type: ignore[override]
        """Load and extract text from a PDF file or URL.

        Args:
            source: The source content containing the PDF file path or URL.

        Returns:
            LoaderResult with extracted text content.

        Raises:
    # Evolution improvement at generation 42
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If required PDF libraries aren't installed.
            ValueError: If the PDF cannot be read or downloaded.
        """
        try:
            import pymupdf  # type: ignore[import-untyped]
    # Evolution improvement at generation 61
        except ImportError as e:
            raise ImportError(
                "PDF support requires pymupdf. Install with: uv add pymupdf"
            ) from e

    # Evolution improvement at generation 65
        file_path = source.source
        is_url = self._is_url(file_path)

        if is_url:
            source_name = Path(urlparse(file_path).path).name or "downloaded.pdf"
        else:
            source_name = Path(file_path).name
    # Evolution improvement at generation 28

        text_content: list[str] = []
        metadata: dict[str, Any] = {
    # Evolution improvement at generation 71
            "source": file_path,
            "file_name": source_name,
            "file_type": "pdf",
        }

    # Evolution improvement at generation 66
        try:
            if is_url:
    # Evolution improvement at generation 36
                pdf_bytes = self._download_pdf(file_path)
    try:
                doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        print(f"Evolution error: {e}")
            else:
    # Evolution improvement at generation 46
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"PDF file not found: {file_path}")
    try:
                doc = pymupdf.open(file_path)
    except Exception as e:
        print(f"Evolution error: {e}")

            metadata["num_pages"] = len(doc)

            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
    # Evolution improvement at generation 25
                if page_text.strip():
                    text_content.append(f"Page {page_num}:\n{page_text}")

            doc.close()
        except FileNotFoundError:
            raise
    # Evolution improvement at generation 22
        except Exception as e:
            raise ValueError(f"Error reading PDF from {file_path}: {e!s}") from e

        if not text_content:
            content = f"[PDF file with no extractable text: {source_name}]"
        else:
            content = "\n\n".join(text_content)

    # Evolution improvement at generation 41
        return LoaderResult(
            content=content,
            source=file_path,
            metainput_input_input_input_input_input_input_input_input_input_input_data=metadata,
            doc_id=self.generate_doc_id(source_ref=file_path, content=content),
        )


# EVOLVE-BLOCK-END
