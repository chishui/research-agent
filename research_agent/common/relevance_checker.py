"""LLM relevance check: is a paper actually about the topic?"""

from __future__ import annotations

import logging
import re

from anthropic import AnthropicBedrock

from .config import ResearchConfig
from .llm import complete
from .paper import Paper

logger = logging.getLogger(__name__)

# One "<n>: YES|NO" verdict per line; tolerates "[n]", "n.", "n -" etc.
_VERDICT = re.compile(r"^\W*(\d+)\W+(YES|NO)\b", re.IGNORECASE | re.MULTILINE)


class RelevanceChecker:
    """Uses Claude to decide whether papers are actually about the topic.

    Runs on the title + abstract (already available from search) so off-topic
    papers are filtered out before the expensive download + parse steps.
    Papers are screened ``relevance_batch_size`` at a time, one call per batch.
    """

    _SYSTEM = (
        "You are a research librarian screening search results. For each "
        "numbered paper, decide whether it is genuinely about the user's "
        "topic, based on its title and abstract. Keyword overlap is not "
        "enough — the paper must actually be about the topic to be relevant. "
        "Respond with exactly one line per paper, in order, of the form "
        "'<number>: YES' or '<number>: NO', and nothing else."
    )

    def __init__(self, client: AnthropicBedrock, config: ResearchConfig):
        self._client = client
        self._config = config

    def is_relevant(self, topic: str, paper: Paper) -> bool:
        return self.check(topic, [paper])[0]

    def check(self, topic: str, papers: list[Paper]) -> list[bool]:
        """Return one relevance verdict per paper, in input order."""
        size = max(1, self._config.relevance_batch_size)
        verdicts: list[bool] = []
        for i in range(0, len(papers), size):
            verdicts.extend(self._check_batch(topic, papers[i : i + size]))
        return verdicts

    def _check_batch(self, topic: str, papers: list[Paper]) -> list[bool]:
        listing = "\n\n".join(
            f"[{n}] Title: {p.title}\n"
            f"Abstract: {p.abstract or '(no abstract available)'}"
            for n, p in enumerate(papers, 1)
        )
        text = complete(
            self._client,
            model=self._config.relevance_model,
            max_tokens=self._config.relevance_max_tokens,
            system=self._SYSTEM,
            user=(
                f"Topic: {topic}\n\nPapers:\n\n{listing}\n\n"
                f"Is each paper about the topic? Answer YES or NO for each."
            ),
        )
        answers = {int(n): v.upper() == "YES" for n, v in _VERDICT.findall(text)}
        if len(papers) == 1 and 1 not in answers:
            answers[1] = text.strip().upper().startswith("YES")

        verdicts = []
        for n, paper in enumerate(papers, 1):
            if n not in answers and len(papers) > 1:
                # A skipped line shouldn't silently discard a paper.
                logger.warning("No verdict for %r in batch; rechecking alone", paper.title)
                verdicts.extend(self._check_batch(topic, [paper]))
                continue
            relevant = answers.get(n, False)
            logger.info(
                "Relevance check for %r: %s",
                paper.title,
                "relevant" if relevant else "not relevant",
            )
            verdicts.append(relevant)
        return verdicts
