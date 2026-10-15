"""Title a finished research report and save it as a Markdown file.

The agents return a report as a Markdown string. This module asks Claude for a
short descriptive title for that report, then writes the report to
``<out_dir>/<slugified-title>.md`` with the title as its top-level heading.

Usage:
    path = write_report(report, topic)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from anthropic import AnthropicBedrock

from common import ResearchConfig, complete

logger = logging.getLogger(__name__)

_TITLE_SYSTEM = (
    "You name research reports. Given the topic and the report, reply with one "
    "title and nothing else — no quotes, no Markdown, no trailing period, at "
    "most 12 words. The title should say what the report concluded, not merely "
    "restate the topic."
)

# Enough of the report for the title to reflect the conclusion, without paying
# for the full paper-by-paper section.
_TITLE_CONTEXT_CHARS = 6000


def generate_title(
    report: str,
    topic: str,
    *,
    client: Optional[AnthropicBedrock] = None,
    config: Optional[ResearchConfig] = None,
) -> str:
    """Ask Claude for a one-line title for ``report``.

    Falls back to the topic if the model call fails or returns nothing — a
    missing title should never lose a completed report.
    """
    config = config or ResearchConfig()
    client = client or AnthropicBedrock()
    user = (
        f"Topic: {topic}\n\n"
        f"Report:\n{report[:_TITLE_CONTEXT_CHARS]}\n\n"
        "Title:"
    )
    try:
        title = complete(
            client,
            model=config.model,
            max_tokens=config.title_max_tokens,
            system=_TITLE_SYSTEM,
            user=user,
        )
    except Exception:
        logger.exception("Title generation failed; falling back to the topic")
        return topic.strip()
    title = _clean_title(title)
    if not title:
        logger.warning("Empty title from the model; falling back to the topic")
        return topic.strip()
    logger.info("Generated title: %s", title)
    return title


def write_report(
    report: str,
    topic: str,
    *,
    out_dir: Optional[Path] = None,
    client: Optional[AnthropicBedrock] = None,
    config: Optional[ResearchConfig] = None,
) -> Path:
    """Title ``report`` and write it to a Markdown file; return the path."""
    config = config or ResearchConfig()
    out_dir = Path(out_dir) if out_dir is not None else config.report_dir
    title = generate_title(report, topic, client=client, config=config)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(out_dir / f"{slugify(title)}.md")
    path.write_text(_with_title(report, title), encoding="utf-8")
    logger.info("Wrote report: %s", path)
    return path


def slugify(title: str, max_length: int = 80) -> str:
    """Turn a title into a filesystem-safe, hyphenated stem."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:max_length]
    return slug.strip("-").lower() or "research-report"


def _clean_title(text: str) -> str:
    """Reduce the model's reply to a single bare line of title text."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    # Quotes and heading markers can appear in either order ('"# Title"').
    line = re.sub(r'^[#\s"\'“”‘’]+|[\s"\'“”‘’]+$', "", line)
    return line.rstrip(".").strip()


def _with_title(report: str, title: str) -> str:
    """Make ``title`` the report's top-level heading.

    The agents emit their own "# Research report: <topic>" heading; replace it
    so the file's heading and its name agree.
    """
    body = report.strip()
    heading = f"# {title}"
    if body.startswith("# "):
        _, _, body = body.partition("\n")
    return f"{heading}\n\n{body.strip()}".rstrip() + "\n"


def _unique_path(path: Path) -> Path:
    """Return ``path``, or ``<stem>-2.md``-style variant if it already exists."""
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find a free filename near {path}")
