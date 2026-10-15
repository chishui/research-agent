"""Shared configuration for the research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Bedrock cross-region inference profile IDs.
DEFAULT_MODEL = "us.anthropic.claude-opus-5-5"
SUMMARY_MODEL = "us.anthropic.claude-opus-4-8"


@dataclass
class ResearchConfig:
    model: str = DEFAULT_MODEL
    summary_model: str = SUMMARY_MODEL
    relevance_model: str = SUMMARY_MODEL
    download_dir: Path = Path("./agent_research_papers")
    report_dir: Path = Path("./research_reports")
    # Thinking counts against max_tokens (Opus 5.5 always thinks), so even
    # one-word answers need headroom. Non-streaming calls must stay under the
    # SDK's ~21K-token cap.
    relevance_max_tokens: int = 2000
    relevance_batch_size: int = 10  # papers screened per model call
    summary_max_tokens: int = 1500
    conclusion_max_tokens: int = 16000
    selector_max_tokens: int = 2000
    title_max_tokens: int = 2000
    request_timeout: float = 60.0
    # How many candidates to fetch per provider page while looking for enough
    # valid papers, and the hard cap on candidates examined per request.
    search_batch_size: int = 10
    max_candidates: int = 100
