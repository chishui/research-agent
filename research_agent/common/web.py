"""General-purpose web/document helpers.

Source-agnostic building blocks (not tied to ``Paper`` or any provider):

  - ``web_search(query)``            -> list of {title, url, snippet}
  - ``fetch_url(url)``               -> readable text of a web page
  - ``download_pdf(url, dest_dir)``  -> local Path of the downloaded PDF
  - ``parse_pdf(path)``              -> markdown text of the PDF

These back the general agent tools in ``tools.py``. They degrade gracefully:
network/availability failures raise ``WebError`` with a clear message rather
than leaking library-specific exceptions, so callers (and the agent) can react.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}


class WebError(RuntimeError):
    """Raised when a web operation fails or a service is unavailable."""


# --------------------------------------------------------------------------- #
# Web search                                                                  #
# --------------------------------------------------------------------------- #
def web_search(query: str, max_results: int = 8, timeout: float = 30.0) -> list[dict]:
    """Search the web and return ``[{title, url, snippet}, ...]``.

    Uses the DuckDuckGo HTML endpoint (no API key). Some networks block this
    behind an anti-bot challenge; in that case a ``WebError`` is raised so the
    caller can fall back to a dedicated source (e.g. an arXiv/PubMed provider).
    """
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=_HEADERS,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise WebError(f"web search request failed: {exc}") from exc

    # DuckDuckGo returns 202 with a challenge page when it blocks automated use.
    if resp.status_code == 202:
        raise WebError(
            "web search is unavailable here (search engine returned an "
            "anti-bot challenge). Use a dedicated paper source instead."
        )
    if not resp.ok:
        raise WebError(f"web search failed with HTTP {resp.status_code}")

    results = _parse_ddg(resp.text, max_results)
    if not results:
        raise WebError(
            "web search returned no parseable results (the search engine may be "
            "blocking automated access). Use a dedicated paper source instead."
        )
    return results


def _parse_ddg(html: str, max_results: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []
    for res in soup.select("div.result, div.web-result"):
        a = res.select_one("a.result__a")
        if not a:
            continue
        snippet_el = res.select_one(".result__snippet")
        results.append(
            {
                "title": a.get_text(" ", strip=True),
                "url": _unwrap_ddg_url(a.get("href", "")),
                "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
            }
        )
        if len(results) >= max_results:
            break
    return results


def _unwrap_ddg_url(href: str) -> str:
    """DuckDuckGo wraps result links as /l/?uddg=<encoded real url>."""
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs:
        return qs["uddg"][0]
    return href


def scholar_search(
    query: str, max_results: int = 8, timeout: float = 30.0
) -> list[dict]:
    """Search Google Scholar; return ``[{title, url, snippet, meta, pdf_url}]``.

    Scholar focuses on academic literature and exposes a PDF link per result
    when one is freely available. It is bot-protected and may serve a CAPTCHA
    under heavy use; in that case a ``WebError`` is raised.
    """
    params = {"q": query, "hl": "en"}
    try:
        resp = requests.get(
            "https://scholar.google.com/scholar",
            params=params,
            headers=_HEADERS,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise WebError(f"scholar search request failed: {exc}") from exc

    body = resp.text.lower()
    if resp.status_code != 200 or "/sorry/" in body or "not a robot" in body:
        raise WebError(
            "Google Scholar is unavailable (rate-limited or CAPTCHA). Use a "
            "dedicated paper source instead."
        )

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict] = []
    for block in soup.select("div.gs_ri"):
        h = block.select_one("h3.gs_rt")
        a = h.select_one("a") if h else None
        title = h.get_text(" ", strip=True) if h else ""
        if not title:
            continue
        meta = block.select_one("div.gs_a")
        snippet = block.select_one("div.gs_rs")
        # The free PDF link, when present, sits in a sibling .gs_ggs block.
        parent = block.find_parent("div", class_="gs_r")
        pdf_a = parent.select_one("div.gs_ggsd a") if parent else None
        results.append(
            {
                "title": title,
                "url": a.get("href", "") if a else "",
                "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                "meta": meta.get_text(" ", strip=True) if meta else "",
                "pdf_url": pdf_a.get("href", "") if pdf_a else "",
            }
        )
        if len(results) >= max_results:
            break

    if not results:
        raise WebError(
            "Google Scholar returned no parseable results (it may be blocking "
            "automated access). Use a dedicated paper source instead."
        )
    return results


# --------------------------------------------------------------------------- #
# Fetch a URL as readable text                                                #
# --------------------------------------------------------------------------- #
def fetch_url(url: str, max_chars: int = 20000, timeout: float = 60.0) -> str:
    """Fetch ``url`` and return its readable text (HTML stripped of markup).

    Truncates to ``max_chars`` to keep results context-window friendly.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise WebError(f"failed to fetch {url}: {exc}") from exc

    ctype = resp.headers.get("Content-Type", "")
    if "html" not in ctype and "xml" not in ctype:
        # Non-HTML (e.g. plain text / json) — return as-is, truncated.
        text = resp.text
    else:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text


# --------------------------------------------------------------------------- #
# Download a PDF                                                              #
# --------------------------------------------------------------------------- #
def download_pdf(
    url: str, dest_dir: Path, timeout: float = 60.0
) -> Path:
    """Download a PDF from ``url`` into ``dest_dir``; return the local path.

    Caches by filename — an already-downloaded, non-empty file is reused.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / _pdf_filename(url)
    if path.exists() and path.stat().st_size > 0 and _looks_like_pdf(path.read_bytes()):
        logger.info("Using cached PDF: %s", path)
        return path
    logger.info("Downloading PDF %s -> %s", url, path)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise WebError(f"failed to download {url}: {exc}") from exc

    content = resp.content
    ctype = resp.headers.get("Content-Type", "").lower()
    # Many "PDF" links actually serve an HTML landing/paywall/redirect page
    # (which PDFium later rejects with a "Data format error"). Detect that here
    # and fail with a clear, actionable message instead of saving a fake PDF.
    if not _looks_like_pdf(content):
        hint = ""
        if "html" in ctype or content[:512].lstrip()[:1] == b"<":
            hint = (
                " The server returned an HTML page, not a PDF (likely a "
                "paywall, login, or landing page). Try fetch_url on the article "
                "page, or use a different source for the full text."
            )
        raise WebError(
            f"{url} did not return a PDF (Content-Type: {ctype or 'unknown'})."
            + hint
        )
    path.write_bytes(content)
    return path


def _looks_like_pdf(data: bytes) -> bool:
    """True if ``data`` begins with the PDF magic header (allowing leading
    whitespace/BOM that some servers prepend)."""
    head = data[:1024].lstrip(b"\xef\xbb\xbf \t\r\n")
    return head.startswith(b"%PDF-")


def _pdf_filename(url: str) -> str:
    tail = urllib.parse.urlparse(url).path.rstrip("/").split("/")[-1] or "download"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", tail)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


# --------------------------------------------------------------------------- #
# Parse a PDF to markdown                                                     #
# --------------------------------------------------------------------------- #
def parse_pdf(path: Path) -> str:
    """Parse a local PDF file to markdown text (via docling)."""
    from .pdf_parser import PDFParser

    path = Path(path)
    if not path.exists():
        raise WebError(f"PDF not found: {path}")
    if not _looks_like_pdf(path.read_bytes()):
        raise WebError(
            f"{path.name} is not a valid PDF (it looks like HTML or another "
            "format). The download likely returned a landing/paywall page."
        )
    try:
        return PDFParser().to_markdown(path)
    except Exception as exc:  # docling/runtime failures
        raise WebError(f"failed to parse PDF {path}: {exc}") from exc
