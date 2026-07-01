"""ResearchAgent on the Strands Agents SDK.

This is the Strands port of ``agent.py``. The hand-written tool-use loop is
replaced by a ``strands.Agent``: we hand it a ``BedrockModel``, the same
``SYSTEM_PROMPT``, and the ``@tool`` methods from ``ResearchTools``. Strands runs
the loop — sending tool schemas, executing tool calls, feeding results back —
until the model stops calling tools and emits the final report.

The system prompt and final-output cleanup are imported from ``agent.py`` so the
two implementations stay behaviorally identical; only the framing changed.

Usage:
    report = StrandsResearchAgent().run("effect of intermittent fasting on health")
"""

from __future__ import annotations

import logging
from typing import Optional

from strands import Agent
from strands.models import BedrockModel

from agent import SYSTEM_PROMPT, _strip_preamble
from common import DEFAULT_MODEL, ResearchConfig
from strands_tools import ResearchTools

logger = logging.getLogger(__name__)


class StrandsResearchAgent:
    """Drives the research tools via the Strands SDK agent loop."""

    def __init__(
        self,
        config: Optional[ResearchConfig] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 10000,
        region_name: Optional[str] = None,
    ):
        self._config = config or ResearchConfig()
        self._model_id = model
        self._max_tokens = max_tokens
        self._region_name = region_name

    def run(self, topic: str) -> str:
        """Research ``topic`` and return the final review report (Markdown)."""
        # Fresh working-set per run: each ResearchTools instance owns its own
        # handle registries (candidates/parsed/summaries/downloads).
        tools = ResearchTools(config=self._config)

        bedrock_kwargs = {
            "model_id": self._model_id,
            "max_tokens": self._max_tokens,
        }
        if self._region_name:
            bedrock_kwargs["region_name"] = self._region_name
        model = BedrockModel(**bedrock_kwargs)

        # callback_handler=None silences Strands' default stdout streaming of
        # the tool-use trajectory, so the only thing written to stdout is our
        # returned report (keeps `> report.md` clean). Progress is still visible
        # via -v logging on the underlying modules.
        agent = Agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=tools.as_tools(),
            callback_handler=None,
        )

        result = agent(f"Write a literature-review report on: {topic}")
        # Strip any preamble before the first Markdown heading; the model often
        # emits "Here is the report:" despite the prompt forbidding it.
        return _strip_preamble(str(result))


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Research-agent report writer (Strands SDK)."
    )
    parser.add_argument("topic", help="Topic or question to research")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(StrandsResearchAgent().run(args.topic))
