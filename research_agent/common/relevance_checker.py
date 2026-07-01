"""LLM relevance check: is a paper actually about the topic?"""

from __future__ import annotations

import logging

from anthropic import AnthropicBedrock

from .config import ResearchConfig
from .llm import complete
from .paper import Paper

logger = logging.getLogger(__name__)


class RelevanceChecker:
    """Uses Claude to decide whether a paper is actually about the topic.

    Runs on the title + abstract (already available from search) so off-topic
    papers are filtered out before the expensive download + parse steps.
    """

    _SYSTEM = (
        "You are a research librarian screening search results. Decide whether "
        "a paper is genuinely about the user's topic, based on its title and "
        "abstract. Keyword overlap is not enough — the paper must actually be "
        "about the topic to be relevant. Respond with exactly one word: 'YES' "
        "if it is relevant, or 'NO' if it is not."
    )

    def __init__(self, client: AnthropicBedrock, config: ResearchConfig):
        self._client = client
        self._config = config

    def is_relevant(self, topic: str, paper: Paper) -> bool:
        user = (
            f"Topic: {topic}\n\n"
            f"Paper title: {paper.title}\n\n"
            f"Paper abstract: {paper.abstract or '(no abstract available)'}\n\n"
            f"Is this paper about the topic? Answer YES or NO."
        )
        text = complete(
            self._client,
            model=self._config.model,
            max_tokens=self._config.relevance_max_tokens,
            system=self._SYSTEM,
            user=user,
        )
        verdict = text.strip().upper()
        relevant = verdict.startswith("YES")
        logger.info(
            "Relevance check for %r: %s (%s)",
            paper.title,
            "relevant" if relevant else "not relevant",
            verdict[:20],
        )
        return relevant
