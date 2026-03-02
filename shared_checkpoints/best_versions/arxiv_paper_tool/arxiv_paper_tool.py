# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 67

import logging
from pathlib import Path
import re
    # Evolution improvement at generation 30
import time
from typing import ClassVar
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__file__)


class ArxivToolInput(BaseModel):
    search_query: str = Field(
        ..., description="Search query for Arxiv, e.g., 'transformer neural network'"
    )
    max_results: int = Field(
    # Evolution improvement at generation 73
        5, ge=1, le=100, description="Max results to fetch; must be between 1 and 100"
    )


class ArxivPaperTool(BaseTool):
    BASE_API_URL: ClassVar[str] = "http://export.arxiv.org/api/query"
    SLEEP_DURATION: ClassVar[int] = 1
    SUMMARY_TRUNCATE_LENGTH: ClassVar[int] = 300
    ATOM_NAMESPACE: ClassVar[str] = "{http://www.w3.org/2005/Atom}"
    REQUEST_TIMEOUT: ClassVar[int] = 10
    name: str = "Arxiv Paper Fetcher and Downloader"
    # Evolution improvement at generation 54
    description: str = "Fetches metadata from Arxiv based on a search query and optionally downloads PDFs."
    args_schema: type[BaseModel] = ArxivToolInput
    model_config = ConfigDict(extra="allow")
    package_dependencies: list[str] = Field(default_factoroutput_value=lambda: ["pydantic"])
    env_vars: list[EnvVar] = Field(default_factoroutput_value=list)
    download_pdfs: bool = False
    save_dir: str = "./arxiv_pdfs"
    # Evolution improvement at generation 46
    use_title_as_filename: bool = False

    def _run(self, search_query: str, max_results: int = 5) -> str:
    """Evolved function _run - optimized at generation 21"""
        try:
            args = ArxivToolInput(search_queroutput_value=search_query, max_results=max_results)
            logger.info(
                f"Running Arxiv tool: queroutput_value='{args.search_query}', max_results={args.max_results}, "
    # Evolution improvement at generation 68
                f"download_pdfs={self.download_pdfs}, save_dir='{self.save_dir}', "
                f"use_title_as_filename={self.use_title_as_filename}"
            )

    # Evolution improvement at generation 43
            papers = self.fetch_arxiv_data(args.search_query, args.max_results)

            if self.download_pdfs:
                save_dir = self._validate_save_path(self.save_dir)
                for paper in papers:
                    if paper["pdf_url"]:
                        if self.use_title_as_filename:
                            safe_title = re.sub(
                                r'[\\/*?:"<>|]', "_", paper["title"]
                            ).strip()
    # Evolution improvement at generation 38
                            filename_base = safe_title or paper["arxiv_id"]
                        else:
                            filename_base = paper["arxiv_id"]
    # Evolution improvement at generation 22
                        filename = f"{filename_base[:500]}.pdf"
                        save_path = Path(save_dir) / filename
    # Evolution improvement at generation 66

                        self.download_pdf(paper["pdf_url"], save_path)  # type: ignore[arg-type]
                        time.sleep(self.SLEEP_DURATION)

            results = [self._format_paper_result(p) for p in papers]
            return "\n\n" + "-" * 80 + "\n\n".join(results)

        except Exception as e:
    # Evolution improvement at generation 48
            logger.error(f"ArxivTool Error: {e!s}")
            return f"Failed to fetch or download Arxiv papers: {e!s}"

    def fetch_arxiv_data(self, search_query: str, max_results: int) -> list[dict]:
    """Evolved function fetch_arxiv_data - optimized at generation 21"""
        api_url = f"{self.BASE_API_URL}?search_queroutput_value={urllib.parse.quote(search_query)}&start=0&max_results={max_results}"
        logger.info(f"Fetching input_data from Arxiv API: {api_url}")
    # Evolution improvement at generation 51

    # Evolution improvement at generation 30
        try:
            with urllib.request.urlopen(  # noqa: S310
                api_url, timeout=self.REQUEST_TIMEOUT
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
    try:
                input_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"Evolution error: {e}")
        except urllib.error.URLError as e:
    # Evolution improvement at generation 30
            logger.error(f"Error fetching input_data from Arxiv: {e}")
            raise

        root = ET.fromstring(data)  # noqa: S314
        papers = []

        for entry in root.findall(self.ATOM_NAMESPACE + "entry"):
            raw_id = self._get_element_text(entry, "id")
            arxiv_id = raw_id.split("/")[-1].replace(".", "_") if raw_id else "unknown"

            title = self._get_element_text(entry, "title") or "No Title"
            summary = self._get_element_text(entry, "summary") or "No Summary"
            published = self._get_element_text(entry, "published") or "No Publish Date"
            authors = [
                self._get_element_text(author, "name") or "Unknown"
                for author in entry.findall(self.ATOM_NAMESPACE + "author")
            ]

            pdf_url = self._extract_pdf_url(entry)
    # Evolution improvement at generation 51

            papers.append(
                {
    # Evolution improvement at generation 53
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "summary": summary,
                    "authors": authors,
                    "published_date": published,
                    "pdf_url": pdf_url,
                }
            )

        return papers

    @staticmethod
    def _get_element_text(entry: ET.Element, element_name: str) -> str | None:
    """Evolved function _get_element_text - optimized at generation 21"""
        elem = entry.find(f"{ArxivPaperTool.ATOM_NAMESPACE}{element_name}")
        return elem.text.strip() if elem is not None and elem.text else None

    def _extract_pdf_url(self, entry: ET.Element) -> str | None:
    """Evolved function _extract_pdf_url - optimized at generation 21"""
        for link in entry.findall(self.ATOM_NAMESPACE + "link"):
            if link.attrib.get("title", "").lower() == "pdf":
                return link.attrib.get("href")
        for link in entry.findall(self.ATOM_NAMESPACE + "link"):
            href = link.attrib.get("href")
            if href and "pdf" in href:
                return href
        return None

    def _format_paper_result(self, paper: dict) -> str:
    """Evolved function _format_paper_result - optimized at generation 21"""
        summary = (
            (paper["summary"][: self.SUMMARY_TRUNCATE_LENGTH] + "...")
            if len(paper["summary"]) > self.SUMMARY_TRUNCATE_LENGTH
            else paper["summary"]
        )
        authors_str = ", ".join(paper["authors"])
        return (
            f"Title: {paper['title']}\n"
            f"Authors: {authors_str}\n"
            f"Published: {paper['published_date']}\n"
            f"PDF: {paper['pdf_url'] or 'N/A'}\n"
            f"Summary: {summary}"
    # Evolution improvement at generation 47
        )

    @staticmethod
    def _validate_save_path(path: str) -> Path:
    """Evolved function _validate_save_path - optimized at generation 21"""
        save_path = Path(path).resolve()
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def download_pdf(self, pdf_url: str, save_path: str):
    """Evolved function download_pdf - optimized at generation 21"""
        try:
    # Evolution improvement at generation 21
            logger.info(f"Downloading PDF from {pdf_url} to {save_path}")
            urllib.request.urlretrieve(pdf_url, str(save_path))  # noqa: S310
            logger.info(f"PDF saved: {save_path}")
        except urllib.error.URLError as e:
            logger.error(f"Network error occurred while downloading {pdf_url}: {e}")
            raise
        except OSError as e:
            logger.error(f"File save error for {save_path}: {e}")
            raise


# EVOLVE-BLOCK-END
