"""Agent tools for the research workflow.

Each stage of the pipeline (provider selection, relevance checking, paper
summarization, synthesis) is exposed as a ``Tool`` the ``ResearchAgent`` can
call. Tools share a ``ToolContext`` that holds the LLM client/config and the
working set of papers — so candidates, fetched papers, and summaries are passed
between tool calls **by handle** (``p1``, ``p2`` …) instead of shipping full
paper bodies through the model's context window.

No agent framework — these are plain classes with a JSON tool schema and a
``run(input, ctx)`` method, driven by the hand-written loop in ``agent.py``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from pathlib import Path

from anthropic import AnthropicBedrock

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
    web_search,
)
from providers import PROVIDERS, get_provider
from research import (
    PaperSummarizer,
    PaperSummary,
    ProviderSelector,
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


# --------------------------------------------------------------------------- #
# Shared context                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ToolContext:
    """Shared state passed to every tool call.

    Holds the LLM client/config plus the working set of papers keyed by a short
    handle (``p1``, ``p2`` …). Tools register papers here and exchange handles,
    keeping large paper bodies out of the model's context.
    """

    client: AnthropicBedrock
    config: ResearchConfig
    candidates: dict[str, Paper] = field(default_factory=dict)
    parsed: dict[str, ParsedPaper] = field(default_factory=dict)
    summaries: dict[str, PaperSummary] = field(default_factory=dict)
    downloads: dict[str, str] = field(default_factory=dict)  # handle -> pdf path
    _counter: int = 0

    def new_handle(self, prefix: str = "d") -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def add_candidate(self, paper: Paper) -> str:
        """Register a candidate paper, returning its handle (dedup by id)."""
        for handle, existing in self.candidates.items():
            if existing.paper_id == paper.paper_id:
                return handle
        self._counter += 1
        handle = f"p{self._counter}"
        self.candidates[handle] = paper
        return handle

    def get(self, handle: str) -> Paper:
        if handle not in self.candidates:
            raise KeyError(f"unknown paper handle {handle!r}")
        return self.candidates[handle]

    def store_text(self, handle: str, text: str) -> None:
        """Attach fetched full text to a candidate so summarize_paper can use it.

        Used by the general (non-provider) fetch path: the body goes in the
        ``introduction`` slot, the candidate's existing abstract is kept.
        """
        paper = self.get(handle)
        self.parsed[handle] = ParsedPaper(
            paper=paper,
            abstract=paper.abstract,
            introduction=text,
            conclusion="",
        )


# --------------------------------------------------------------------------- #
# Tool base class                                                             #
# --------------------------------------------------------------------------- #
class Tool(ABC):
    """A single capability the agent can invoke."""

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    @classmethod
    def definition(cls) -> dict:
        """The Anthropic tool schema for this tool."""
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls.input_schema,
        }

    @abstractmethod
    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        """Execute the tool and return a compact text result for the model."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Tool: select providers                                                      #
# --------------------------------------------------------------------------- #
class SelectProvidersTool(Tool):
    """Wraps ``ProviderSelector`` — choose paper sources for a topic."""

    name = "select_providers"
    description = (
        "Given a research topic, choose which academic paper source(s) to "
        "search (e.g. arxiv for CS/physics, pubmed for biomedicine). Returns "
        "the provider names whose coverage fits the topic. Call this first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The research topic to find sources for.",
            }
        },
        "required": ["topic"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        topic = tool_input["topic"]
        selector = ProviderSelector(ctx.client, ctx.config)
        names = selector.select_many(topic)
        return "Selected providers: " + ", ".join(names)


# --------------------------------------------------------------------------- #
# Tool: search a provider for candidates                                      #
# --------------------------------------------------------------------------- #
class SearchPapersTool(Tool):
    """Search one provider for candidate papers (metadata only, no fetch)."""

    name = "search_papers"
    description = (
        "Search one paper source for candidates matching the topic. Returns a "
        "numbered list of candidate papers (handle, source, title, and a short "
        "abstract preview). Does NOT download full papers — use fetch_paper for "
        "that. Call once per selected provider."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": "Provider name from select_providers (e.g. 'arxiv').",
            },
            "topic": {
                "type": "string",
                "description": "The research topic to search for.",
            },
            "limit": {
                "type": "integer",
                "description": "Max candidates to return.",
                "default": 8,
            },
        },
        "required": ["provider", "topic"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        provider = tool_input["provider"]
        topic = tool_input["topic"]
        limit = int(tool_input.get("limit", 8))
        source = get_provider(provider, config=ctx.config, client=ctx.client)
        candidates = source.find_candidates(topic, limit)
        if not candidates:
            return f"No candidates found in {provider} for {topic!r}."
        lines = [f"Found {len(candidates)} candidates in {provider}:"]
        for paper in candidates:
            handle = ctx.add_candidate(paper)
            preview = (paper.abstract or "(no abstract)")[:240]
            lines.append(f"- {handle} [{paper.source}] {paper.title}")
            lines.append(f"    {preview}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool: relevance check                                                       #
# --------------------------------------------------------------------------- #
class CheckRelevanceTool(Tool):
    """Wraps ``RelevanceChecker`` — is a candidate actually about the topic?"""

    name = "check_relevance"
    description = (
        "Judge whether a candidate paper (by handle) is genuinely about the "
        "topic, based on its title and abstract. Use to filter out off-topic "
        "candidates before fetching/summarizing them. Accepts multiple handles."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Candidate handles to check (e.g. ['p1','p2']).",
            },
            "topic": {
                "type": "string",
                "description": "The research topic to check relevance against.",
            },
        },
        "required": ["paper_ids", "topic"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        topic = tool_input["topic"]
        handles = tool_input["paper_ids"]
        checker = RelevanceChecker(ctx.client, ctx.config)
        lines = []
        for handle in handles:
            try:
                paper = ctx.get(handle)
            except KeyError as exc:
                lines.append(f"{handle}: error ({exc})")
                continue
            relevant = checker.is_relevant(topic, paper)
            lines.append(f"{handle}: {'relevant' if relevant else 'not relevant'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool: fetch (download + parse) a paper                                      #
# --------------------------------------------------------------------------- #
class FetchPaperTool(Tool):
    """Download + parse a candidate's full text (abstract/intro/conclusion)."""

    name = "fetch_paper"
    description = (
        "Download and parse a candidate paper (by handle) to extract its "
        "abstract, introduction, and conclusion. Required before summarize_paper. "
        "Accepts multiple handles. Reports which sections were extracted."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Candidate handles to fetch (e.g. ['p1','p2']).",
            }
        },
        "required": ["paper_ids"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        handles = tool_input["paper_ids"]
        lines = []
        for handle in handles:
            try:
                paper = ctx.get(handle)
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
                    paper.source, config=ctx.config, client=ctx.client
                )
                parsed = source.fetch(paper)
            except Exception as exc:
                lines.append(f"{handle}: fetch failed ({exc})")
                continue
            if not parsed.has_content():
                lines.append(f"{handle}: no usable sections extracted")
                continue
            ctx.parsed[handle] = parsed
            lines.append(
                f"{handle}: fetched — abstract {len(parsed.abstract)} chars, "
                f"introduction {len(parsed.introduction)} chars, "
                f"conclusion {len(parsed.conclusion)} chars"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool: summarize a paper                                                     #
# --------------------------------------------------------------------------- #
class SummarizePaperTool(Tool):
    """Wraps ``PaperSummarizer`` — summarize one fetched paper's core idea."""

    name = "summarize_paper"
    description = (
        "Summarize the core idea of a fetched paper (by handle): problem, key "
        "idea/method, main results, and why it matters. The paper must have "
        "been fetched first. Accepts multiple handles. Returns each summary."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Fetched paper handles to summarize.",
            }
        },
        "required": ["paper_ids"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        handles = tool_input["paper_ids"]
        summarizer = PaperSummarizer(ctx.client, ctx.config)
        out = []
        for handle in handles:
            parsed = ctx.parsed.get(handle)
            if parsed is None:
                out.append(f"{handle}: not fetched yet — call fetch_paper first")
                continue
            try:
                summary = summarizer.summarize(parsed)
            except Exception as exc:
                out.append(f"{handle}: summarization failed ({exc})")
                continue
            ctx.summaries[handle] = summary
            out.append(f"### {handle}: {parsed.paper.title}\n{summary.summary}")
        return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# Tool: aggregate summaries into a conclusion                                 #
# --------------------------------------------------------------------------- #
class AggregateSummariesTool(Tool):
    """Wraps ``ResearchAggregator`` — synthesize summaries into a conclusion."""

    name = "aggregate_summaries"
    description = (
        "Synthesize the summarized papers into a single coherent conclusion "
        "about the topic: common themes, agreements/disagreements, notable "
        "findings, and open questions. Call after summarizing the papers you "
        "want included. Omit paper_ids to use all summarized papers."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The research topic being synthesized.",
            },
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Summarized paper handles to include. Omit for all of them."
                ),
            },
        },
        "required": ["topic"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        topic = tool_input["topic"]
        handles = tool_input.get("paper_ids") or list(ctx.summaries.keys())
        summaries = [ctx.summaries[h] for h in handles if h in ctx.summaries]
        if not summaries:
            return "No summaries available to aggregate. Summarize papers first."
        aggregator = ResearchAggregator(ctx.client, ctx.config)
        return aggregator.aggregate(topic, summaries)


# --------------------------------------------------------------------------- #
# General-purpose tools (not tied to the paper providers)                     #
# --------------------------------------------------------------------------- #
class WebSearchTool(Tool):
    """General web search (DuckDuckGo, no API key)."""

    name = "web_search"
    description = (
        "Search the general web for a query. Returns a list of results with "
        "title, URL, and snippet. Use for background/context or to find pages "
        "and PDFs; for academic papers prefer the dedicated paper sources. May "
        "be unavailable on some networks."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The web search query."},
            "max_results": {
                "type": "integer",
                "description": "Max results to return.",
                "default": 8,
            },
        },
        "required": ["query"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        query = tool_input["query"]
        max_results = int(tool_input.get("max_results", 8))
        try:
            results = web_search(query, max_results=max_results)
        except WebError as exc:
            return f"web_search unavailable: {exc}"
        lines = [f"Found {len(results)} results for {query!r}:"]
        for r in results:
            lines.append(f"- {r['title']}\n    {r['url']}\n    {r['snippet']}")
        return "\n".join(lines)


class ScholarSearchTool(Tool):
    """Academic web search via Google Scholar."""

    name = "scholar_search"
    description = (
        "Search Google Scholar for academic papers across all fields and "
        "publishers. Returns title, link, authors/venue/year, a snippet, and a "
        "direct PDF link when one is freely available. Good for broad academic "
        "discovery or sources not covered by the dedicated providers (arxiv, "
        "pubmed). Pass any returned pdf_url to pdf_download."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The academic search query."},
            "max_results": {
                "type": "integer",
                "description": "Max results to return.",
                "default": 8,
            },
        },
        "required": ["query"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        query = tool_input["query"]
        max_results = int(tool_input.get("max_results", 8))
        try:
            results = scholar_search(query, max_results=max_results)
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
            handle = ctx.add_candidate(paper)
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


class FetchUrlTool(Tool):
    """Fetch a URL and return its readable text."""

    name = "fetch_url"
    description = (
        "Fetch a web page by URL and return its readable text content (markup "
        "stripped, truncated for length). Use this to read a non-arxiv/pubmed "
        "paper found via scholar_search (pass its handle as paper_id so it can "
        "be summarized), or any page cited elsewhere."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch."},
            "paper_id": {
                "type": "string",
                "description": (
                    "Optional candidate handle (e.g. 'p3'). If given, the fetched "
                    "text is attached to that paper so summarize_paper can use it."
                ),
            },
            "max_chars": {
                "type": "integer",
                "description": "Truncate the returned text to this many chars.",
                "default": 20000,
            },
        },
        "required": ["url"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        url = tool_input["url"]
        paper_id = tool_input.get("paper_id")
        max_chars = int(tool_input.get("max_chars", 20000))
        try:
            text = fetch_url(url, max_chars=max_chars)
        except WebError as exc:
            return f"fetch_url failed: {exc}"
        if paper_id:
            try:
                ctx.store_text(paper_id, text)
                return (
                    f"Fetched {len(text)} chars and attached to {paper_id}; "
                    f"ready for summarize_paper.\n\n{text}"
                )
            except KeyError as exc:
                return f"(could not attach: {exc})\n\n{text}"
        return text


class PdfDownloadTool(Tool):
    """Download a PDF from a URL into the working directory."""

    name = "pdf_download"
    description = (
        "Download a PDF from a URL to local storage. Returns a download handle "
        "(e.g. 'd1') to pass to pdf_parse. Use for PDFs found on the web that "
        "aren't already provided by a paper source."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL of the PDF to download."}
        },
        "required": ["url"],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        url = tool_input["url"]
        try:
            path = download_pdf(url, Path(ctx.config.download_dir))
        except WebError as exc:
            return f"pdf_download failed: {exc}"
        handle = ctx.new_handle("d")
        ctx.downloads[handle] = str(path)
        size = Path(path).stat().st_size
        return f"Downloaded to handle {handle} ({size} bytes). Use pdf_parse({handle})."


class PdfParseTool(Tool):
    """Parse a downloaded PDF to markdown text."""

    name = "pdf_parse"
    description = (
        "Parse a downloaded PDF (by download handle from pdf_download, or by a "
        "local file path) into markdown text. Returns the document text, "
        "truncated for length."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "handle": {
                "type": "string",
                "description": "Download handle from pdf_download (e.g. 'd1').",
            },
            "path": {
                "type": "string",
                "description": "Alternatively, a local path to a PDF file.",
            },
            "paper_id": {
                "type": "string",
                "description": (
                    "Optional candidate handle (e.g. 'p3'). If given, the parsed "
                    "text is attached to that paper so summarize_paper can use it."
                ),
            },
            "max_chars": {
                "type": "integer",
                "description": "Truncate the returned markdown to this many chars.",
                "default": 20000,
            },
        },
        "required": [],
    }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        handle = tool_input.get("handle")
        path = tool_input.get("path")
        paper_id = tool_input.get("paper_id")
        if handle:
            path = ctx.downloads.get(handle)
            if path is None:
                return f"Unknown download handle {handle!r}. Call pdf_download first."
        if not path:
            return "Provide either a download 'handle' or a 'path'."
        max_chars = int(tool_input.get("max_chars", 20000))
        try:
            markdown = parse_pdf(Path(path))
        except WebError as exc:
            return f"pdf_parse failed: {exc}"
        attached = ""
        if paper_id:
            try:
                ctx.store_text(paper_id, markdown)
                attached = f"Attached to {paper_id}; ready for summarize_paper.\n\n"
            except KeyError as exc:
                attached = f"(could not attach: {exc})\n\n"
        if len(markdown) > max_chars:
            markdown = markdown[:max_chars] + "\n…[truncated]"
        return attached + markdown


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
ALL_TOOLS: list[Tool] = [
    # Discovery: Google Scholar is the single entry point. It registers each
    # result as a candidate handle and tags its source (arxiv/pubmed/web).
    ScholarSearchTool(),
    # Screening + source-routed retrieval:
    CheckRelevanceTool(),
    FetchPaperTool(),   # for [arxiv]/[pubmed] handles (structured provider parse)
    FetchUrlTool(),     # for [web] handles (attach text via paper_id)
    PdfDownloadTool(),  # + pdf_parse, for [web] handles backed by a PDF
    PdfParseTool(),
    # Synthesis:
    SummarizePaperTool(),
    AggregateSummariesTool(),
    # Not offered to the agent in the Scholar-first flow:
    #   SelectProvidersTool / SearchPapersTool — no pre-selection; Scholar leads.
    #   WebSearchTool — Scholar is the discovery tool; general web is blocked here.
]


def tool_definitions() -> list[dict]:
    """Anthropic tool schemas for every research tool."""
    return [t.definition() for t in ALL_TOOLS]


def tool_registry() -> dict[str, Tool]:
    """Map of tool name -> tool instance, for dispatch in the agent loop."""
    return {t.name: t for t in ALL_TOOLS}
