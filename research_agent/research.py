"""Research tool for the agent in main.py.

Given a topic, this module:
  0. Selects which paper provider/source to use for the topic    (ProviderSelector)
  1. Searches the chosen provider, which finds relevant papers
     and returns them already downloaded + parsed                 (providers/*)
  2. Summarizes each paper's core idea with Claude               (PaperSummarizer)
  3. Aggregates the per-paper summaries into a single conclusion
     about the topic                                             (ResearchAggregator)

Reusable building blocks (data types, config, the LLM helper, PDF download +
parsing, relevance checking) live in the ``common`` package. Paper sources live
in the ``providers`` package — each source is its own module, owns its own
fetch + parse, and is exposed as an agent tool (with a schema describing the
realms it covers). ``ProviderSelector`` lets Claude pick the right source for a
topic before any searching happens.

Entry points:
  - ``Researcher(...).research(topic, top_n)``  -> ``ResearchReport``
  - ``run_research_tool({...})``                -> str   (for agent tool use)
  - ``TOOL_DEFINITION``                         -> Anthropic tool schema

Extra dependency: ``pip install docling`` (used only for PDF parsing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from anthropic import AnthropicBedrock

from common import (
    DEFAULT_MODEL,
    Paper,
    ParsedPaper,
    RelevanceChecker,
    ResearchConfig,
    complete,
)
from providers import (
    default_provider,
    get_provider,
    get_provider_tools,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data model                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class PaperSummary:
    """An LLM-generated summary of a single paper's idea."""

    paper: Paper
    summary: str


@dataclass
class ResearchReport:
    """Final output: per-paper summaries plus an aggregated conclusion."""

    topic: str
    providers: list[str] = field(default_factory=list)
    summaries: list[PaperSummary] = field(default_factory=list)
    conclusion: str = ""

    def format(self) -> str:
        lines = [f"# Research report: {self.topic}", ""]
        if self.providers:
            label = "Sources" if len(self.providers) > 1 else "Source"
            lines.append(f"_{label}: {', '.join(self.providers)}_")
            lines.append("")
        lines.append("## Conclusion")
        lines.append(self.conclusion or "(no conclusion generated)")
        lines.append("")
        lines.append("## Papers reviewed")
        if not self.summaries:
            lines.append("(none)")
        for i, s in enumerate(self.summaries, 1):
            authors = ", ".join(s.paper.authors[:3])
            if len(s.paper.authors) > 3:
                authors += " et al."
            src = f"{s.paper.source}: " if s.paper.source else ""
            lines.append("")
            lines.append(f"### {i}. {s.paper.title}")
            lines.append(
                f"_{src}{authors} ({s.paper.published[:10]}) — {s.paper.pdf_url}_"
            )
            lines.append("")
            lines.append(s.summary)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Stage 0: select a provider                                                  #
# --------------------------------------------------------------------------- #
class ProviderSelector:
    """Uses Claude tool-use to pick paper provider(s) for a topic.

    Each provider exposes a tool schema (name + description + the realms it
    covers). We present them to Claude and let it choose which to call; the
    chosen tool name(s) are the provider(s) we search with.

    ``select`` returns the single best provider; ``select_many`` returns every
    provider whose realms fit the topic (e.g. a topic spanning CS *and* biology
    might select both ``arxiv`` and ``pubmed``).
    """

    _SYSTEM_ONE = (
        "You select the best academic paper source for a research topic. You "
        "are given several paper-source tools, each describing the research "
        "areas it covers. Call exactly one tool — the source most appropriate "
        "for the topic. Do not answer in text; just make the tool call."
    )

    _SYSTEM_MANY = (
        "You select academic paper sources for a research topic. You are given "
        "several paper-source tools, each describing the research areas it "
        "covers. Prefer the most specialized source(s) for the topic. Call a "
        "source only if a SUBSTANTIAL body of directly relevant work genuinely "
        "lives there — not merely because the topic is adjacent to one of its "
        "areas. Most topics need just one source; call several only when the "
        "topic truly spans multiple fields (e.g. a method from one field "
        "applied to another). When unsure about a second source, leave it out. "
        "Do not answer in text; just make the tool calls."
    )

    def __init__(self, client: AnthropicBedrock, config: ResearchConfig):
        self._client = client
        self._config = config
        self._tools = get_provider_tools()

    def select(self, topic: str) -> str:
        """Return the single best provider name (falls back to the default)."""
        if len(self._tools) == 1:
            name = self._tools[0]["name"]
            logger.info("Only one provider available; using %r", name)
            return name
        names = self._call(topic, self._SYSTEM_ONE, allow_parallel=False)
        if names:
            logger.info("Selected provider %r for %r", names[0], topic)
            return names[0]
        fallback = default_provider().name
        logger.info("Falling back to default provider %r", fallback)
        return fallback

    def select_many(
        self, topic: str, max_providers: Optional[int] = None
    ) -> list[str]:
        """Return every provider whose realms fit the topic (1 or more).

        Falls back to the single best provider (then the default) if the model
        selects nothing. ``max_providers`` optionally caps how many are kept.
        """
        if len(self._tools) == 1:
            return [self._tools[0]["name"]]
        names = self._call(topic, self._SYSTEM_MANY, allow_parallel=True)
        if not names:
            names = [self.select(topic)]
        if max_providers is not None:
            names = names[:max_providers]
        logger.info("Selected providers %s for %r", names, topic)
        return names

    def _call(
        self, topic: str, system: str, allow_parallel: bool
    ) -> list[str]:
        """Run provider-selection tool-use; return chosen tool names in order.

        Dedupes while preserving first-seen order. Returns [] on error so the
        caller can fall back.
        """
        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.selector_max_tokens,
                system=system,
                tools=self._tools,
                tool_choice={
                    "type": "any",
                    "disable_parallel_tool_use": not allow_parallel,
                },
                messages=[{"role": "user", "content": f"Topic: {topic}"}],
            )
        except Exception:
            logger.exception("Provider selection failed")
            return []
        seen: set[str] = set()
        names: list[str] = []
        for block in response.content:
            if block.type == "tool_use" and block.name not in seen:
                seen.add(block.name)
                names.append(block.name)
        return names


# --------------------------------------------------------------------------- #
# Stage 2: summarize one paper                                                #
# --------------------------------------------------------------------------- #
class PaperSummarizer:
    """Uses Claude to summarize the core idea of a single paper."""

    _SYSTEM = (
        "You are a research assistant. Given the abstract, introduction, and "
        "conclusion of a paper, summarize its core idea for someone surveying "
        "a topic. Cover: (1) the problem addressed, (2) the key idea or method, "
        "(3) the main results, and (4) why it matters. Be precise and concise; "
        "do not invent details that are not supported by the provided text."
    )

    def __init__(self, client: AnthropicBedrock, config: ResearchConfig):
        self._client = client
        self._config = config

    def summarize(self, parsed: ParsedPaper) -> PaperSummary:
        logger.info("Summarizing: %s", parsed.paper.title)
        content = self._build_prompt(parsed)
        text = complete(
            self._client,
            model=self._config.model,
            max_tokens=self._config.summary_max_tokens,
            system=self._SYSTEM,
            user=content,
        )
        return PaperSummary(paper=parsed.paper, summary=text.strip())

    @staticmethod
    def _build_prompt(parsed: ParsedPaper) -> str:
        parts = [f"Title: {parsed.paper.title}", ""]
        for label, body in (
            ("Abstract", parsed.abstract),
            ("Introduction", parsed.introduction),
            ("Conclusion", parsed.conclusion),
        ):
            if body:
                parts.append(f"## {label}")
                parts.append(body)
                parts.append("")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Stage 3: aggregate                                                          #
# --------------------------------------------------------------------------- #
class ResearchAggregator:
    """Aggregates per-paper summaries into a single conclusion on the topic."""

    _SYSTEM = (
        "You are a research analyst. You are given a topic and summaries of "
        "several papers about it. Synthesize them into a coherent conclusion: "
        "identify common themes, key approaches, points of agreement and "
        "disagreement, notable findings, and open questions or gaps. Ground "
        "every claim in the provided summaries and cite papers by their number."
    )

    def __init__(self, client: AnthropicBedrock, config: ResearchConfig):
        self._client = client
        self._config = config

    def aggregate(self, topic: str, summaries: list[PaperSummary]) -> str:
        logger.info("Aggregating %d summaries for %r", len(summaries), topic)
        content = self._build_prompt(topic, summaries)
        return complete(
            self._client,
            model=self._config.model,
            max_tokens=self._config.conclusion_max_tokens,
            system=self._SYSTEM,
            user=content,
        ).strip()

    @staticmethod
    def _build_prompt(topic: str, summaries: list[PaperSummary]) -> str:
        parts = [f"Topic: {topic}", "", "Paper summaries:", ""]
        for i, s in enumerate(summaries, 1):
            parts.append(f"[{i}] {s.paper.title}")
            parts.append(s.summary)
            parts.append("")
        parts.append(
            "Write a synthesized conclusion about the topic based on the "
            "summaries above."
        )
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
class Researcher:
    """Wires provider selection + summarize + aggregate into one pipeline.

    Search/fetch/parse now live inside the provider; the orchestrator passes
    the relevance check in as the provider's ``accept`` predicate and handles
    only summarization and synthesis.
    """

    def __init__(
        self,
        client: Optional[AnthropicBedrock] = None,
        config: Optional[ResearchConfig] = None,
    ):
        self._config = config or ResearchConfig()
        self._client = client or AnthropicBedrock()
        self._selector = ProviderSelector(self._client, self._config)
        self._relevance = RelevanceChecker(self._client, self._config)
        self._summarizer = PaperSummarizer(self._client, self._config)
        self._aggregator = ResearchAggregator(self._client, self._config)

    def research(
        self, topic: str, top_n: int = 5, provider: Optional[str] = None
    ) -> ResearchReport:
        # Stage 0: decide which source(s) to use. By default we select every
        # provider whose realms fit the topic — cross-disciplinary topics are
        # common, and `select_many` naturally collapses to a single source when
        # only one fits (or only one is registered). An explicit `provider`
        # forces just that one.
        if provider:
            provider_names = [provider]
        else:
            provider_names = self._selector.select_many(topic)

        report = ResearchReport(topic=topic, providers=provider_names)

        # Stage 1: each selected provider searches, filters (via our relevance
        # check), fetches, and parses — returning ready-to-summarize papers.
        # We request top_n from each source so every one is well-represented.
        parsed_papers: list[ParsedPaper] = []
        for name in provider_names:
            try:
                source = get_provider(
                    name, config=self._config, client=self._client
                )
                parsed_papers.extend(
                    source.search(
                        topic,
                        top_n=top_n,
                        accept=lambda paper: self._relevance.is_relevant(
                            topic, paper
                        ),
                    )
                )
            except Exception:
                logger.exception("Provider %r failed; skipping", name)

        # Stage 2: summarize each paper (across all sources).
        for parsed in parsed_papers:
            try:
                report.summaries.append(self._summarizer.summarize(parsed))
            except Exception:
                logger.exception("Failed to summarize: %s", parsed.paper.title)

        # Stage 3: synthesize an overall conclusion over the combined set.
        if report.summaries:
            report.conclusion = self._aggregator.aggregate(topic, report.summaries)
        else:
            report.conclusion = (
                "No papers could be processed for this topic, so no conclusion "
                "could be drawn."
            )
        return report


# --------------------------------------------------------------------------- #
# Agent tool integration                                                      #
# --------------------------------------------------------------------------- #
TOOL_DEFINITION = {
    "name": "research_topic",
    "description": (
        "Research a topic by selecting the academic paper source(s) whose "
        "coverage fits it, finding the most relevant papers, reading their "
        "abstract/introduction/conclusion, summarizing each, and synthesizing "
        "an overall conclusion."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The research topic or question to investigate.",
            },
            "top_n": {
                "type": "integer",
                "description": "How many of the most relevant papers to review.",
                "default": 5,
            },
            "max_candidates": {
                "type": "integer",
                "description": (
                    "Maximum number of papers to examine while looking for "
                    "`top_n` relevant ones. Caps the search so a narrow topic "
                    "can't be searched indefinitely."
                ),
                "default": ResearchConfig.max_candidates,
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional paper source to force (e.g. 'arxiv'). If omitted, "
                    "every source whose coverage fits the topic is searched "
                    "automatically."
                ),
            },
        },
        "required": ["topic"],
    },
}


def run_research_tool(tool_input: dict) -> str:
    """Execute the research tool from an agent ``tool_use`` block.

    ``tool_input`` is the ``input`` dict Claude provides for the tool call.
    Returns a formatted report string suitable as a ``tool_result``.
    """
    topic = tool_input["topic"]
    top_n = int(tool_input.get("top_n", 5))
    provider = tool_input.get("provider")
    config = ResearchConfig()
    if "max_candidates" in tool_input:
        config.max_candidates = int(tool_input["max_candidates"])
    return (
        Researcher(config=config)
        .research(topic, top_n=top_n, provider=provider)
        .format()
    )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Research a topic via Claude.")
    parser.add_argument("topic", help="Topic or question to research")
    parser.add_argument("-n", "--top-n", type=int, default=5, help="Number of papers")
    parser.add_argument(
        "-m",
        "--max-candidates",
        type=int,
        default=ResearchConfig.max_candidates,
        help="Max papers to examine while looking for top-n relevant ones",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default=None,
        help=(
            "Force a single paper source (e.g. 'arxiv'); default searches every "
            "source whose coverage fits the topic"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = ResearchConfig(max_candidates=args.max_candidates)
    report = Researcher(config=config).research(
        args.topic, top_n=args.top_n, provider=args.provider
    )
    print(report.format())
