"""ResearchAgent — a tool-using agent that writes paper-review reports.

Unlike ``Researcher`` (a fixed pipeline), the agent decides its own trajectory.
It leads with Google Scholar to discover papers, then routes each result by its
source tag — [arxiv]/[pubmed] go through the provider tool (fetch_paper) for
clean structured text, while [web] sources are read with fetch_url (or
pdf_download + pdf_parse). It screens with check_relevance, summarizes, and
aggregates, driving itself through a hand-written tool-use loop — no agent
framework.

Usage:
    report = ResearchAgent().run("effect of intermittent fasting on health")
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from anthropic import AnthropicBedrock

from common import DEFAULT_MODEL, ResearchConfig
from tools import ToolContext, tool_definitions, tool_registry

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a research assistant that produces literature-review reports on a topic \
by finding, reading, and synthesizing academic papers. You can only access \
papers through the tools below — never rely on memory for specific papers, \
findings, dates, authors, or numbers.

# Tools

- scholar_search(query) — Google Scholar across all fields and publishers. This \
is your entry point for finding papers. Each result is registered as a \
candidate with a handle (p1, p2, …) and a tagged source shown in brackets: \
[arxiv], [pubmed], or [web]. Results include a link and, when free, a pdf.
- check_relevance(paper_ids, topic) — screen candidate handles; keep only the \
ones genuinely about the topic.
- fetch_paper(paper_ids) — fetch + parse candidates whose source is a dedicated \
provider, i.e. [arxiv] or [pubmed]. Gives the cleanest structured text. Only \
works for those sources.
- fetch_url(url, paper_id) — read a [web] paper's page as text. Pass the \
candidate's handle as paper_id so the text is attached for summarizing. Use \
this for any source that is NOT [arxiv]/[pubmed].
- pdf_download(url) then pdf_parse(handle, paper_id) — alternative for a [web] \
candidate that has a free pdf link: download the pdf, then parse it, passing \
the candidate's handle as paper_id.
- summarize_paper(paper_ids) — summarize each candidate that has been fetched \
(via fetch_paper, fetch_url, or pdf_parse).
- aggregate_summaries(topic) — synthesize the summaries into one conclusion.

# Requirements
  - At least 10 distinct, credible sources, each cited as [n]. This is a \
FLOOR, not a target — keep searching and fetching if the topic still has \
uncovered subareas, conflicting findings, or important gaps. Conversely, \
stop only when adding another paper would not meaningfully change the \
conclusion (the same themes keep recurring across new sources).
  - Prefer sources from high-impact-factor journals (e.g. Nature, Science, \
NEJM, Cell, The Lancet, PNAS, JAMA, BMJ) over lower-tier venues. When \
sources disagree on a finding, weight the higher-impact-factor source more \
heavily in your conclusion. Treat preprints (e.g. [arxiv]) as provisional \
unless corroborated by peer-reviewed work.
  - Sections: Background, Key Findings, Open Questions, References.
  - Do not fabricate sources or URLs.
  - Markdown output.

# How to work

1. scholar_search(topic) to discover candidate papers.
2. check_relevance on the returned handles; keep only the on-topic ones. \
Don't anchor on a fixed count — keep going (more scholar_search calls, \
different queries) if the topic still has unexplored subareas; stop when \
new sources are repeating what you already have.
3. Fetch each relevant candidate, ROUTING BY ITS SOURCE TAG:
   - [arxiv] or [pubmed]  → fetch_paper (you may batch several handles).
   - [web]                → fetch_url(url, paper_id=<handle>); or, if it has a \
free pdf link, pdf_download(pdf_url) then pdf_parse(handle, paper_id=<handle>).
4. summarize_paper on the fetched handles.
5. aggregate_summaries(topic).

Be resilient — tools return error strings instead of raising. If fetching or \
summarizing one paper fails, continue with the others; never abort the whole \
review. If pdf_download reports the link was a paywall/HTML page, fall back to \
fetch_url on the article page (or rely on the abstract) rather than retrying \
the same pdf.

# Final output

When you have an aggregated conclusion and per-paper summaries, STOP calling \
tools and emit the report. Your response must START with the character '#' — \
there is no introduction, no "here it is", no closing remarks. Use exactly \
this Markdown structure:

# Research report: <topic>

## Conclusion
<the aggregated synthesis>

## Papers reviewed
### 1. <title>
_<authors> (<date>) — <url>_

<that paper's summary>

(repeat for each reviewed paper)

Do not fabricate papers, findings, citations, or URLs — use only what the \
tools returned."""


class ResearchAgent:
    """Drives the research tools through a manual tool-use loop."""

    def __init__(
        self,
        client: Optional[AnthropicBedrock] = None,
        config: Optional[ResearchConfig] = None,
        model: str = DEFAULT_MODEL,
        max_steps: int = 100,
        max_tokens: int = 10000,
    ):
        self._config = config or ResearchConfig()
        self._client = client or AnthropicBedrock()
        self._model = model
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._tools = tool_definitions()
        self._registry = tool_registry()

    def run(self, topic: str) -> str:
        """Research ``topic`` and return the final review report (Markdown)."""
        ctx = ToolContext(client=self._client, config=self._config)
        messages: list[dict] = [
            {
                "role": "user",
                "content": f"Write a literature-review report on: {topic}"
            }
        ]

        for step in range(1, self._max_steps + 1):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                tools=self._tools,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                # No more tool calls — the agent's text is the final report.
                # Strip any preamble before the first Markdown heading; the
                # model often emits "Here is the report:" despite the prompt
                # forbidding it (and this Bedrock model doesn't allow
                # assistant-prefill to prevent it structurally).
                final = _strip_preamble(_text_of(response.content))
                logger.info("Agent finished after %d step(s)", step)
                return final

            messages.append({"role": "assistant", "content": response.content})

            # Execute every tool_use block and return the results together.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                logger.info("Tool call: %s(%s)", block.name, _short(block.input))
                result = self._dispatch(block.name, block.input, ctx)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        logger.warning("Agent hit max_steps (%d) without finishing", self._max_steps)
        # Best effort: return whatever text the agent last produced.
        return _text_of(messages[-1].get("content", [])) or (
            "Research did not complete within the step budget."
        )

    def _dispatch(self, name: str, tool_input: dict, ctx: ToolContext) -> str:
        """Run one tool, converting any error into a tool result the model can
        recover from (rather than crashing the loop)."""
        tool = self._registry.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}."
        try:
            return tool.run(tool_input, ctx)
        except Exception as exc:  # surface the error back to the model
            logger.exception("Tool %r raised", name)
            return f"Error running {name}: {exc}"


def _text_of(content) -> str:
    """Concatenate text blocks from a message content list."""
    if isinstance(content, str):
        return content
    return "".join(
        b.text if hasattr(b, "text") else b.get("text", "")
        for b in content
        if (getattr(b, "type", None) or (isinstance(b, dict) and b.get("type")))
        == "text"
    )


def _short(value, limit: int = 120) -> str:
    s = json.dumps(value, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def _strip_preamble(text: str) -> str:
    """Drop anything before the first Markdown heading.

    The model occasionally emits a "Here is the report:" line before the
    actual '#'. Slice from the first '#' that starts a line; if there is
    no heading, return the text unchanged.
    """
    idx = text.find("\n#")
    if idx >= 0:
        return text[idx + 1 :].lstrip()
    if text.lstrip().startswith("#"):
        return text.lstrip()
    return text


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Research-agent report writer.")
    parser.add_argument("topic", help="Topic or question to research")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(ResearchAgent().run(args.topic))
