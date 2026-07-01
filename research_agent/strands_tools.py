"""Research tools for the Strands-SDK agent.

This is the Strands port of ``tools.py``. The hand-written tool-use loop and the
``Tool``-subclass schemas are gone; instead each capability is a ``@tool``-
decorated method on ``ResearchTools``. Strands builds the JSON schema from each
method's signature, type hints, and docstring, and drives the tool-use loop.

Why a class instead of free functions: the workflow passes papers between tool
calls **by handle** (``p1``, ``p2`` …) to keep large paper bodies out of the
model's context. That working set is shared, mutable state. Strands' built-in
``agent.state`` only stores JSON-serializable values, but our handles map to
live ``Paper``/``ParsedPaper``/``PaperSummary`` objects — so we follow the
Strands "class-based tools" pattern and hold the registries as instance
attributes. Every ``@tool`` method reads/writes the same ``self`` state, exactly
as the old ``ToolContext`` did.

The tool *bodies* still call the existing, tested logic in ``research.py`` /
``common`` / ``providers`` through an ``AnthropicBedrock`` client — only the
agent framing changed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from anthropic import AnthropicBedrock
from strands import tool

from common import (
    Paper,
    ParsedPaper,
    RelevanceChecker,
    ResearchConfig,
    WebError,
    download_pdf,
    fetch_url,
    parse_pdf,
    scholar_search,
)
from providers import PROVIDERS, get_provider
from research import (
    PaperSummarizer,
    PaperSummary,
    ResearchAggregator,
)

logger = logging.getLogger(__name__)

# Map URL substrings to the provider that should handle that source. Used to
# route a Scholar result to a specific provider tool when one fits.
_SOURCE_HINTS = {
    "arxiv.org": "arxiv",
    "pubmed.ncbi.nlm.nih.gov": "pubmed",
    "ncbi.nlm.nih.gov": "pubmed",
}


def detect_source(*urls: str) -> str:
    """Return the provider name that owns any of ``urls``, else 'web'.

    Only returns a provider that is actually registered, so routing never
    points the agent at a tool that can't handle it.
    """
    for url in urls:
        u = (url or "").lower()
        for hint, name in _SOURCE_HINTS.items():
            if hint in u and name in PROVIDERS:
                return name
    return "web"


class ResearchTools:
    """Holds the shared working set and exposes it as Strands tools.

    The registries below replace the old ``ToolContext`` dataclass: candidates,
    fetched/parsed papers, summaries, and PDF downloads are all keyed by short
    handles (``p1``, ``p2`` … for papers; ``d1``, ``d2`` … for downloads).
    Construct one instance per research run and pass its bound ``@tool`` methods
    to the ``Agent``.
    """

    def __init__(
        self,
        client: Optional[AnthropicBedrock] = None,
        config: Optional[ResearchConfig] = None,
    ):
        self.client = client or AnthropicBedrock()
        self.config = config or ResearchConfig()
        self.candidates: dict[str, Paper] = {}
        self.parsed: dict[str, ParsedPaper] = {}
        self.summaries: dict[str, PaperSummary] = {}
        self.downloads: dict[str, str] = {}  # handle -> pdf path
        self._counter = 0

    # ------------------------------------------------------------------ #
    # Internal handle bookkeeping (ported from ToolContext)              #
    # ------------------------------------------------------------------ #
    def _new_handle(self, prefix: str = "d") -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def _add_candidate(self, paper: Paper) -> str:
        """Register a candidate paper, returning its handle (dedup by id)."""
        for handle, existing in self.candidates.items():
            if existing.paper_id == paper.paper_id:
                return handle
        self._counter += 1
        handle = f"p{self._counter}"
        self.candidates[handle] = paper
        return handle

    def _get(self, handle: str) -> Paper:
        if handle not in self.candidates:
            raise KeyError(f"unknown paper handle {handle!r}")
        return self.candidates[handle]

    def _store_text(self, handle: str, text: str) -> None:
        """Attach fetched full text to a candidate so summarize can use it.

        Used by the general (non-provider) fetch path: the body goes in the
        ``introduction`` slot, the candidate's existing abstract is kept.
        """
        paper = self._get(handle)
        self.parsed[handle] = ParsedPaper(
            paper=paper,
            abstract=paper.abstract,
            introduction=text,
            conclusion="",
        )

    # ------------------------------------------------------------------ #
    # Discovery                                                          #
    # ------------------------------------------------------------------ #
    @tool
    def scholar_search(self, query: str, max_results: int = 8) -> str:
        """Search Google Scholar for academic papers across all fields and
        publishers. Returns title, link, authors/venue/year, a snippet, and a
        direct PDF link when one is freely available. This is the entry point
        for finding papers; each result is registered as a candidate with a
        handle (p1, p2, ...) and a tagged source ([arxiv], [pubmed], or [web]).

        Args:
            query: The academic search query.
            max_results: Max results to return.
        """
        try:
            results = scholar_search(query, max_results=int(max_results))
        except WebError as exc:
            return f"scholar_search unavailable: {exc}"
        lines = [f"Found {len(results)} Scholar results for {query!r}:"]
        for r in results:
            source = detect_source(r["url"], r.get("pdf_url", ""))
            # Register as a candidate so it gets a handle the rest of the
            # workflow (check_relevance, fetch_*, summarize) can refer to.
            paper = Paper(
                paper_id=r["url"],
                title=r["title"],
                authors=[],
                published=r.get("meta", ""),
                abstract=r.get("snippet", ""),
                pdf_url=r.get("pdf_url", "") or r["url"],
                source=source,
            )
            handle = self._add_candidate(paper)
            lines.append(f"- {handle} [{source}] {r['title']}")
            if r.get("meta"):
                lines.append(f"    {r['meta']}")
            lines.append(f"    link: {r['url']}")
            if r.get("pdf_url"):
                lines.append(f"    pdf: {r['pdf_url']}")
            if r.get("snippet"):
                lines.append(f"    {r['snippet']}")
        lines.append(
            "\nNext: check_relevance on the handles, then fetch — use "
            "fetch_paper for [arxiv]/[pubmed] handles, and fetch_url (or "
            "pdf_download+pdf_parse for a PDF) for others."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Screening                                                          #
    # ------------------------------------------------------------------ #
    @tool
    def check_relevance(self, paper_ids: list[str], topic: str) -> str:
        """Judge whether candidate papers (by handle) are genuinely about the
        topic, based on each title and abstract. Use to filter out off-topic
        candidates before fetching/summarizing them.

        Args:
            paper_ids: Candidate handles to check (e.g. ['p1','p2']).
            topic: The research topic to check relevance against.
        """
        checker = RelevanceChecker(self.client, self.config)
        lines = []
        for handle in paper_ids:
            try:
                paper = self._get(handle)
            except KeyError as exc:
                lines.append(f"{handle}: error ({exc})")
                continue
            relevant = checker.is_relevant(topic, paper)
            lines.append(f"{handle}: {'relevant' if relevant else 'not relevant'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Source-routed retrieval                                            #
    # ------------------------------------------------------------------ #
    @tool
    def fetch_paper(self, paper_ids: list[str]) -> str:
        """Download and parse candidate papers (by handle) whose source is a
        dedicated provider — i.e. [arxiv] or [pubmed] — extracting abstract,
        introduction, and conclusion. Gives the cleanest structured text.
        Required before summarize_paper. Only works for those sources; use
        fetch_url (or pdf_download + pdf_parse) for [web] sources.

        Args:
            paper_ids: Candidate handles to fetch (e.g. ['p1','p2']).
        """
        lines = []
        for handle in paper_ids:
            try:
                paper = self._get(handle)
            except KeyError as exc:
                lines.append(f"{handle}: {exc}")
                continue
            if paper.source not in PROVIDERS:
                lines.append(
                    f"{handle}: source {paper.source!r} has no provider tool — "
                    "use fetch_url (or pdf_download + pdf_parse) instead."
                )
                continue
            try:
                source = get_provider(
                    paper.source, config=self.config, client=self.client
                )
                parsed = source.fetch(paper)
            except Exception as exc:
                lines.append(f"{handle}: fetch failed ({exc})")
                continue
            if not parsed.has_content():
                lines.append(f"{handle}: no usable sections extracted")
                continue
            self.parsed[handle] = parsed
            lines.append(
                f"{handle}: fetched — abstract {len(parsed.abstract)} chars, "
                f"introduction {len(parsed.introduction)} chars, "
                f"conclusion {len(parsed.conclusion)} chars"
            )
        return "\n".join(lines)

    @tool
    def fetch_url(self, url: str, paper_id: str = "", max_chars: int = 20000) -> str:
        """Fetch a web page by URL and return its readable text content (markup
        stripped, truncated for length). Use this to read a non-arxiv/pubmed
        ([web]) paper found via scholar_search, or any page cited elsewhere.

        Args:
            url: The URL to fetch.
            paper_id: Optional candidate handle (e.g. 'p3'). If given, the
                fetched text is attached to that paper so summarize_paper can
                use it.
            max_chars: Truncate the returned text to this many chars.
        """
        try:
            text = fetch_url(url, max_chars=int(max_chars))
        except WebError as exc:
            return f"fetch_url failed: {exc}"
        if paper_id:
            try:
                self._store_text(paper_id, text)
                return (
                    f"Fetched {len(text)} chars and attached to {paper_id}; "
                    f"ready for summarize_paper.\n\n{text}"
                )
            except KeyError as exc:
                return f"(could not attach: {exc})\n\n{text}"
        return text

    @tool
    def pdf_download(self, url: str) -> str:
        """Download a PDF from a URL to local storage. Returns a download handle
        (e.g. 'd1') to pass to pdf_parse. Use for PDFs found on the web that
        aren't already provided by a paper source.

        Args:
            url: URL of the PDF to download.
        """
        try:
            path = download_pdf(url, Path(self.config.download_dir))
        except WebError as exc:
            return f"pdf_download failed: {exc}"
        handle = self._new_handle("d")
        self.downloads[handle] = str(path)
        size = Path(path).stat().st_size
        return f"Downloaded to handle {handle} ({size} bytes). Use pdf_parse({handle})."

    @tool
    def pdf_parse(
        self,
        handle: str = "",
        path: str = "",
        paper_id: str = "",
        max_chars: int = 20000,
    ) -> str:
        """Parse a downloaded PDF (by download handle from pdf_download, or by a
        local file path) into markdown text. Returns the document text,
        truncated for length.

        Args:
            handle: Download handle from pdf_download (e.g. 'd1').
            path: Alternatively, a local path to a PDF file.
            paper_id: Optional candidate handle (e.g. 'p3'). If given, the
                parsed text is attached to that paper so summarize_paper can
                use it.
            max_chars: Truncate the returned markdown to this many chars.
        """
        if handle:
            resolved = self.downloads.get(handle)
            if resolved is None:
                return f"Unknown download handle {handle!r}. Call pdf_download first."
            path = resolved
        if not path:
            return "Provide either a download 'handle' or a 'path'."
        try:
            markdown = parse_pdf(Path(path))
        except WebError as exc:
            return f"pdf_parse failed: {exc}"
        attached = ""
        if paper_id:
            try:
                self._store_text(paper_id, markdown)
                attached = f"Attached to {paper_id}; ready for summarize_paper.\n\n"
            except KeyError as exc:
                attached = f"(could not attach: {exc})\n\n"
        if len(markdown) > int(max_chars):
            markdown = markdown[: int(max_chars)] + "\n…[truncated]"
        return attached + markdown

    # ------------------------------------------------------------------ #
    # Synthesis                                                          #
    # ------------------------------------------------------------------ #
    @tool
    def summarize_paper(self, paper_ids: list[str]) -> str:
        """Summarize the core idea of fetched papers (by handle): problem, key
        idea/method, main results, and why it matters. Each paper must have been
        fetched first (via fetch_paper, fetch_url, or pdf_parse).

        Args:
            paper_ids: Fetched paper handles to summarize.
        """
        summarizer = PaperSummarizer(self.client, self.config)
        out = []
        for handle in paper_ids:
            parsed = self.parsed.get(handle)
            if parsed is None:
                out.append(f"{handle}: not fetched yet — call fetch_paper first")
                continue
            try:
                summary = summarizer.summarize(parsed)
            except Exception as exc:
                out.append(f"{handle}: summarization failed ({exc})")
                continue
            self.summaries[handle] = summary
            out.append(f"### {handle}: {parsed.paper.title}\n{summary.summary}")
        return "\n\n".join(out)

    @tool
    def aggregate_summaries(self, topic: str, paper_ids: Optional[list[str]] = None) -> str:
        """Synthesize the summarized papers into a single coherent conclusion
        about the topic: common themes, agreements/disagreements, notable
        findings, and open questions. Call after summarizing the papers you want
        included.

        Args:
            topic: The research topic being synthesized.
            paper_ids: Summarized paper handles to include. Omit for all of them.
        """
        handles = paper_ids or list(self.summaries.keys())
        summaries = [self.summaries[h] for h in handles if h in self.summaries]
        if not summaries:
            return "No summaries available to aggregate. Summarize papers first."
        aggregator = ResearchAggregator(self.client, self.config)
        return aggregator.aggregate(topic, summaries)

    # ------------------------------------------------------------------ #
    # Registration                                                       #
    # ------------------------------------------------------------------ #
    def as_tools(self) -> list:
        """The bound ``@tool`` methods to hand to ``Agent(tools=...)``.

        Order mirrors the old ``ALL_TOOLS`` registry: discovery → screening →
        source-routed retrieval → synthesis.
        """
        return [
            self.scholar_search,
            self.check_relevance,
            self.fetch_paper,
            self.fetch_url,
            self.pdf_download,
            self.pdf_parse,
            self.summarize_paper,
            self.aggregate_summaries,
        ]
