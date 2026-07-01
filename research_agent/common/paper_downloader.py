"""Downloads a paper's PDF to a local cache directory."""

from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path

from .paper import Paper

logger = logging.getLogger(__name__)

_USER_AGENT = "agent-research-tool/1.0 (+https://example.com)"


class PaperDownloader:
    """Downloads a paper's PDF to a local cache directory."""

    def __init__(self, dest_dir: Path, timeout: float = 60.0):
        self._dest_dir = Path(dest_dir)
        self._timeout = timeout

    def download(self, paper: Paper) -> Path:
        self._dest_dir.mkdir(parents=True, exist_ok=True)
        path = self._dest_dir / f"{safe_filename(paper)}.pdf"
        if path.exists() and path.stat().st_size > 0:
            logger.info("Using cached PDF: %s", path)
            paper.pdf_path = path
            return path
        logger.info("Downloading %s -> %s", paper.pdf_url, path)
        req = urllib.request.Request(
            paper.pdf_url, headers={"User-Agent": _USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = resp.read()
        path.write_bytes(data)
        paper.pdf_path = path
        return path


def safe_filename(paper: Paper) -> str:
    """Build a filesystem-safe, source-prefixed filename for a paper."""
    # arXiv ids look like http://arxiv.org/abs/1234.5678v1 -> 1234.5678v1
    tail = paper.paper_id.rstrip("/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", tail) or "paper"
    prefix = f"{paper.source}_" if paper.source else ""
    return f"{prefix}{name}"
