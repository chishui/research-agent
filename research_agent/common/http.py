"""Shared HTTP helper for providers that scrape HTML pages.

Uses a ``requests`` session with browser-like headers. Note: sites behind a
Cloudflare/Akamai JS challenge (e.g. JAMA Network) cannot be fetched with plain
HTTP from a non-browser client — they return 403 until a browser solves the
challenge. For those, inject a browser-backed fetcher (see ``HtmlClient``).
"""

from __future__ import annotations

import requests

# Browser-like headers — many publisher sites reject the default agent.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
}


class HtmlClient:
    """Fetches HTML pages over HTTP.

    Wrapping the fetch behind a small class lets a caller swap in a different
    implementation (e.g. a headless-browser fetcher that can solve a Cloudflare
    challenge) without changing the providers that depend on it.
    """

    def __init__(self, timeout: float = 60.0, headers: dict | None = None):
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(headers or BROWSER_HEADERS)

    def get(self, url: str, params: dict | None = None) -> str:
        """GET ``url`` and return the response body as text (raises on error)."""
        resp = self._session.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.text
