"""Shared building blocks for the research pipeline.

Foundational, dependency-light pieces used by both the ``providers`` package
and the top-level ``research`` orchestration:

  - ``Paper`` / ``ParsedPaper``     — shared data types
  - ``ResearchConfig`` / ``DEFAULT_MODEL`` — configuration
  - ``complete``                    — single non-streaming Claude call
  - ``PaperDownloader``             — PDF download + caching
  - ``PDFParser`` / ``extract_sections`` — docling parse + section extraction
  - ``RelevanceChecker``            — LLM "is this about the topic?" check
"""

from __future__ import annotations

from .config import DEFAULT_MODEL, ResearchConfig
from .http import HtmlClient
from .llm import complete
from .paper import Paper
from .paper_downloader import PaperDownloader, safe_filename
from .parsed_paper import ParsedPaper
from .pdf_parser import PDFParser, extract_sections
from .relevance_checker import RelevanceChecker
from .web import (
    WebError,
    download_pdf,
    fetch_url,
    parse_pdf,
    scholar_search,
    web_search,
)

__all__ = [
    "DEFAULT_MODEL",
    "ResearchConfig",
    "HtmlClient",
    "complete",
    "Paper",
    "ParsedPaper",
    "PaperDownloader",
    "safe_filename",
    "PDFParser",
    "extract_sections",
    "RelevanceChecker",
    "WebError",
    "web_search",
    "scholar_search",
    "fetch_url",
    "download_pdf",
    "parse_pdf",
]
