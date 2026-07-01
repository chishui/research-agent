"""Base class for paper providers.

A *paper provider* wraps one external paper source (arXiv, PubMed, ...). Each
provider:

  - declares the research realms it covers (``name`` / ``description`` /
    ``realms``) so an agent can pick the right source for a topic, and
  - exposes ``search(topic, top_n, ...)`` returning a list of ``ParsedPaper`` —
    i.e. the provider owns fetching *and* parsing. PDF-backed sources download
    and run docling; HTML-only sources can override ``_fetch`` to scrape the
    landing page instead.

The shared ``search`` template handles pagination, the relevance filter, and
the ``top_n`` / ``max_candidates`` caps; a subclass only implements
``_search_page`` (cheap metadata lookup) and, if it isn't PDF-backed, overrides
``_fetch``.

The ``Paper`` / ``ParsedPaper`` data types live in the ``common`` package.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from anthropic import AnthropicBedrock

from common import DEFAULT_MODEL, Paper, ParsedPaper, PaperDownloader, PDFParser

logger = logging.getLogger(__name__)

# Predicate the orchestrator passes in to decide whether a candidate is worth
# fetching (e.g. an LLM relevance check). Defaults to "accept everything".
AcceptFn = Callable[[Paper], bool]


class PaperProvider(ABC):
    """Base class for a single paper source, usable as an agent tool."""

    #: Short, stable identifier used as the tool name (e.g. ``"arxiv"``).
    name: str = ""
    #: One-line description of what this source covers — shown to the agent.
    description: str = ""
    #: The research areas/realms this source is good for. Surfaced in the tool
    #: schema so the agent can match a topic to the right provider.
    realms: tuple[str, ...] = ()
    #: Whether this provider is usable. Set False for sources that can't be
    #: reached from this environment (e.g. JAMA, which is behind a Cloudflare
    #: JS challenge) so they're excluded from selection rather than failing
    #: mid-run. The code stays for when a browser-capable fetcher is available.
    enabled: bool = True

    def __init__(
        self,
        timeout: float = 60.0,
        download_dir: Path = Path("./agent_research_papers"),
        search_batch_size: int = 10,
        max_candidates: int = 100,
        client: Optional[AnthropicBedrock] = None,
        model: str = DEFAULT_MODEL,
    ):
        self._timeout = timeout
        self._batch_size = search_batch_size
        self._max_candidates = max_candidates
        # Shared fetch/parse machinery for PDF-backed providers.
        self._downloader = PaperDownloader(download_dir, timeout=timeout)
        self._parser = PDFParser()
        # Optional LLM client for providers that rewrite the search query.
        # Lazily created so providers that don't need it pay nothing.
        self._client = client
        self._model = model

    @property
    def client(self) -> AnthropicBedrock:
        """Lazily-created Bedrock client, shared by query-rewriting providers."""
        if self._client is None:
            self._client = AnthropicBedrock()
        return self._client

    # ----------------------------------------------------------------- #
    # Public API                                                        #
    # ----------------------------------------------------------------- #
    def search(
        self,
        topic: str,
        top_n: int = 5,
        accept: Optional[AcceptFn] = None,
    ) -> list[ParsedPaper]:
        """Return up to ``top_n`` parsed papers relevant to ``topic``.

        Pages through results until ``top_n`` papers pass ``accept`` and parse
        successfully, or until ``max_candidates`` papers have been examined.
        Skipped/failed papers don't count toward ``top_n``.
        """
        accept = accept or (lambda _paper: True)
        results: list[ParsedPaper] = []
        start = 0
        examined = 0

        # Translate the user's topic into this source's query syntax once.
        # ``accept`` still receives the original topic for relevance checks.
        query = self._build_query(topic)
        if query != topic:
            logger.info("[%s] rewrote query %r -> %r", self.name, topic, query)

        while len(results) < top_n and examined < self._max_candidates:
            candidates = self._search_page(query, self._batch_size, start=start)
            if not candidates:
                break  # source has no more results
            start += len(candidates)

            for paper in candidates:
                if len(results) >= top_n or examined >= self._max_candidates:
                    break
                examined += 1
                try:
                    if not accept(paper):
                        logger.info("Skipping off-topic paper: %s", paper.title)
                        continue
                    parsed = self._fetch(paper)
                    if not parsed.has_content():
                        logger.warning(
                            "No usable sections extracted: %s", paper.title
                        )
                        continue
                    results.append(parsed)
                except Exception:  # keep going even if one paper fails
                    logger.exception("Skipping paper due to error: %s", paper.title)

        if len(results) < top_n:
            logger.warning(
                "[%s] collected only %d/%d papers for %r after examining %d "
                "candidates (max_candidates=%d)",
                self.name,
                len(results),
                top_n,
                topic,
                examined,
                self._max_candidates,
            )
        else:
            logger.info(
                "[%s] collected %d/%d papers for %r (examined %d candidates)",
                self.name,
                len(results),
                top_n,
                topic,
                examined,
            )
        return results

    def find_candidates(self, topic: str, limit: int) -> list[Paper]:
        """Return up to ``limit`` candidate papers (metadata only, no fetch).

        Like ``search`` but without the relevance filter or PDF/HTML fetch — it
        just pages the source for candidate metadata (title + abstract). Useful
        when a caller (e.g. the agent) wants to inspect/relevance-check
        candidates itself before deciding which to fetch.
        """
        query = self._build_query(topic)
        if query != topic:
            logger.info("[%s] rewrote query %r -> %r", self.name, topic, query)

        results: list[Paper] = []
        start = 0
        while len(results) < limit and start < self._max_candidates:
            page = self._search_page(query, self._batch_size, start=start)
            if not page:
                break
            start += len(page)
            results.extend(page)
        return results[:limit]

    def fetch(self, paper: Paper) -> ParsedPaper:
        """Fetch + parse a single candidate into a ``ParsedPaper`` (public)."""
        return self._fetch(paper)

    # ----------------------------------------------------------------- #
    # Subclass hooks                                                    #
    # ----------------------------------------------------------------- #
    def _build_query(self, topic: str) -> str:
        """Translate the user's topic into this source's query syntax.

        Default is identity — the topic is passed through unchanged. Providers
        whose APIs need specific syntax (boolean operators, field targeting,
        phrase quoting) override this, typically with an LLM rewrite.
        """
        return topic

    @abstractmethod
    def _search_page(
        self, query: str, max_results: int, start: int = 0
    ) -> list[Paper]:
        """Return one page of candidate ``Paper`` metadata (no fetch/parse).

        ``query`` is already in this source's syntax (see ``_build_query``).
        Implementations paginate via ``start`` and stamp each returned
        ``Paper.source`` with ``self.name``.
        """
        raise NotImplementedError

    def _fetch(self, paper: Paper) -> ParsedPaper:
        """Fetch and parse a single candidate into a ``ParsedPaper``.

        Default implementation downloads the PDF and parses it with docling.
        HTML-only sources should override this to scrape the landing page.
        """
        self._downloader.download(paper)
        return self._parser.parse(paper)

    # ----------------------------------------------------------------- #
    # Tool schema                                                       #
    # ----------------------------------------------------------------- #
    @classmethod
    def tool_schema(cls) -> dict:
        """Render this provider as an Anthropic tool definition.

        The agent uses the returned schema to decide whether this source fits
        the topic.
        """
        realms = ", ".join(cls.realms) if cls.realms else "general"
        return {
            "name": cls.name,
            "description": (
                f"{cls.description} Best for these areas: {realms}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The research topic or question to search.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "How many relevant papers to review.",
                        "default": 5,
                    },
                },
                "required": ["topic"],
            },
        }


# --------------------------------------------------------------------------- #
# Helpers shared by provider implementations                                  #
# --------------------------------------------------------------------------- #
def clean(value: str) -> str:
    """Collapse whitespace in a metadata string."""
    return re.sub(r"\s+", " ", value).strip()
