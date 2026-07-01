"""PDF parsing (via docling) and markdown section extraction.

``PDFParser`` turns a downloaded PDF into a ``ParsedPaper``. The markdown
heading -> section logic is also exposed as the module-level ``extract_sections``
helper so HTML-based providers can reuse it on their own markdown/text.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .paper import Paper
from .parsed_paper import ParsedPaper

logger = logging.getLogger(__name__)

# Heading keywords -> the section we map them to.
_SECTION_PATTERNS = {
    "abstract": re.compile(r"^\s*abstract\b", re.IGNORECASE),
    "introduction": re.compile(r"^\s*(?:\d+\.?\s*)?introduction\b", re.IGNORECASE),
    "conclusion": re.compile(
        r"^\s*(?:\d+\.?\s*)?(?:conclusion|conclusions|"
        r"concluding\s+remarks|conclusion\s+and\s+future\s+work)\b",
        re.IGNORECASE,
    ),
}


class PDFParser:
    """Parses a PDF with docling and extracts abstract/intro/conclusion."""

    def __init__(self):
        self._converter = None  # lazily created; docling is an optional dep

    def _get_converter(self):
        if self._converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import (
                    DocumentConverter,
                    PdfFormatOption,
                )
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "docling is required for PDF parsing. Install it with "
                    "`pip install docling`."
                ) from exc
            # arXiv PDFs are born-digital (they have a real text layer), so OCR
            # is unnecessary. Disabling it avoids initializing docling's OCR
            # engine (which can fail to load its models) and speeds parsing up.
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        return self._converter

    def to_markdown(self, pdf_path) -> str:
        """Convert a PDF file to markdown text (docling). Source-agnostic."""
        logger.info("Parsing PDF with docling: %s", pdf_path)
        result = self._get_converter().convert(str(pdf_path))
        return result.document.export_to_markdown()

    def parse(self, paper: Paper) -> ParsedPaper:
        if not paper.pdf_path:
            raise ValueError("paper.pdf_path is not set; download the PDF first")
        markdown = self.to_markdown(paper.pdf_path)
        sections = extract_sections(markdown)
        # The source's abstract is more reliable than heading-scraping, so
        # prefer it when available.
        abstract = sections.get("abstract") or paper.abstract
        return ParsedPaper(
            paper=paper,
            abstract=abstract,
            introduction=sections.get("introduction", ""),
            conclusion=sections.get("conclusion", ""),
        )


def extract_sections(markdown: str) -> dict[str, str]:
    """Walk markdown headings and capture text under matching sections."""
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in markdown.splitlines():
        heading = _heading_text(line)
        if heading is not None:
            current = _classify(heading)
            if current:
                sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
        if "".join(lines).strip()
    }


def _heading_text(line: str) -> Optional[str]:
    m = re.match(r"^\s*#{1,6}\s+(.*)$", line)
    return m.group(1).strip() if m else None


def _classify(heading: str) -> Optional[str]:
    for name, pattern in _SECTION_PATTERNS.items():
        if pattern.match(heading):
            return name
    return None
