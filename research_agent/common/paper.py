"""The shared ``Paper`` metadata type."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Paper:
    """Metadata for a single paper, before any PDF/HTML processing.

    Shared across all providers so downstream stages (parse, summarize) are
    source-agnostic.
    """

    paper_id: str
    title: str
    authors: list[str]
    published: str
    abstract: str  # source-provided abstract (reliable fallback for parsing)
    pdf_url: str
    source: str = ""  # name of the provider that produced this paper
    pdf_path: Optional[Path] = None
