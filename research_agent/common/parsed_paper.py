"""The ``ParsedPaper`` type: a paper plus its extracted sections."""

from __future__ import annotations

from dataclasses import dataclass

from .paper import Paper


@dataclass
class ParsedPaper:
    """A paper with the three target sections extracted from its PDF/HTML."""

    paper: Paper
    abstract: str
    introduction: str
    conclusion: str

    def has_content(self) -> bool:
        return any((self.abstract, self.introduction, self.conclusion))
