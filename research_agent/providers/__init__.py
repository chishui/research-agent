"""Paper provider registry.

Each provider wraps one paper source and is exposed as an agent tool. This
package vends:

  - ``PROVIDERS``            — mapping of provider name -> provider class
  - ``get_provider_tools()`` — list of Anthropic tool schemas, one per provider,
                               for an agent to choose from given a topic
  - ``get_provider(name)``   — instantiate a provider by its tool name

To add a new source, drop a ``providers/<source>.py`` defining a
``PaperProvider`` subclass and register it in ``PROVIDERS`` below.
"""

from __future__ import annotations

from typing import Optional

from anthropic import AnthropicBedrock

from common import ResearchConfig

from .arxiv import ArxivProvider
from .base import PaperProvider
from .jama import JamaProvider
from .pubmed import PubMedProvider

# Registry of available providers, keyed by their tool name.
PROVIDERS: dict[str, type[PaperProvider]] = {
    ArxivProvider.name: ArxivProvider,
    JamaProvider.name: JamaProvider,
    PubMedProvider.name: PubMedProvider,
}


def enabled_providers() -> dict[str, type[PaperProvider]]:
    """Registry entries that are usable in this environment (``enabled``)."""
    return {n: c for n, c in PROVIDERS.items() if c.enabled}


def get_provider_tools() -> list[dict]:
    """Return the tool schemas the agent picks from to select a paper source.

    Only *enabled* providers are offered, so the selector never picks a source
    that can't be reached (e.g. JAMA behind a Cloudflare challenge).
    """
    return [cls.tool_schema() for cls in enabled_providers().values()]


def _kwargs(
    config: Optional[ResearchConfig], client: Optional[AnthropicBedrock]
) -> dict:
    kwargs: dict = {}
    if config is not None:
        kwargs.update(
            timeout=config.request_timeout,
            download_dir=config.download_dir,
            search_batch_size=config.search_batch_size,
            max_candidates=config.max_candidates,
            model=config.model,
        )
    if client is not None:
        kwargs["client"] = client
    return kwargs


def get_provider(
    name: str,
    config: Optional[ResearchConfig] = None,
    client: Optional[AnthropicBedrock] = None,
) -> PaperProvider:
    """Instantiate a provider by its tool name (e.g. ``"arxiv"``)."""
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown provider {name!r}. Available: {sorted(PROVIDERS)}"
        ) from None
    return cls(**_kwargs(config, client))


def default_provider(
    config: Optional[ResearchConfig] = None,
    client: Optional[AnthropicBedrock] = None,
) -> PaperProvider:
    """Return a sensible default provider (arXiv)."""
    return ArxivProvider(**_kwargs(config, client))


__all__ = [
    "PaperProvider",
    "PROVIDERS",
    "enabled_providers",
    "get_provider_tools",
    "get_provider",
    "default_provider",
    "ArxivProvider",
    "JamaProvider",
    "PubMedProvider",
]
