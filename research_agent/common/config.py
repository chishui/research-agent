"""Shared configuration for the research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Matches the client/model convention already used in main.py (Bedrock).
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"


@dataclass
class ResearchConfig:
    model: str = DEFAULT_MODEL
    download_dir: Path = Path("./agent_research_papers")
    relevance_max_tokens: int = 10
    summary_max_tokens: int = 1500
    conclusion_max_tokens: int = 10000
    selector_max_tokens: int = 200
    request_timeout: float = 60.0
    # How many candidates to fetch per provider page while looking for enough
    # valid papers, and the hard cap on candidates examined per request.
    search_batch_size: int = 10
    max_candidates: int = 100
