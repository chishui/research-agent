"""arXiv paper provider.

Wraps the arXiv Atom API. No third-party deps — uses the standard library for
HTTP and XML.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

from common import Paper, complete

from .base import PaperProvider, clean

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_USER_AGENT = "agent-research-tool/1.0 (+https://example.com)"


class ArxivProvider(PaperProvider):
    """Searches arXiv and returns the most relevant papers for a topic."""

    name = "arxiv"
    description = (
        "Searches arXiv, the open-access preprint server for physics, math, "
        "computer science, and quantitative fields."
    )
    realms = (
        "computer science",
        "machine learning / AI",
        "mathematics",
        "physics",
        "statistics",
        "computational/quantitative biology (modeling and theory, NOT clinical "
        "or medical research)",
        "quantitative finance",
        "electrical engineering",
    )

    # System prompt for turning a colloquial topic into an arXiv query string.
    _QUERY_SYSTEM = (
        "You convert a research topic into a search query for the arXiv API. "
        "arXiv matches metadata (title/abstract), so use scholarly terminology "
        "and follow these rules:\n"
        "- Combine the essential concepts with AND, each prefixed with a field "
        "(use 'all:' unless a tighter field is clearly better).\n"
        "- Quote multi-word phrases that should stay together, e.g. "
        'all:"graph neural network".\n'
        "- DROP meta-noise that does not appear in scholarly metadata: 'sota', "
        "'state-of-the-art', 'state of the art', 'best', 'novel', 'latest', "
        "'using', 'approach', 'method', 'algorithm' (unless central), 'paper', "
        "'research'.\n"
        "- Keep it focused: 2-4 concepts is usually ideal. Don't over-constrain.\n"
        "- Expand obvious acronyms only when helpful, keeping both forms with "
        'OR, e.g. (all:"approximate nearest neighbor" OR all:ANN).\n'
        "Respond with ONLY the query string, nothing else."
    )

    def _build_query(self, topic: str) -> str:
        """Rewrite the topic into an arXiv ``search_query`` value.

        Uses the LLM to produce proper arXiv syntax (field-prefixed, AND-joined,
        phrase-quoted, noise dropped). Falls back to AND-joining the topic's
        words if the LLM is unavailable or returns something unusable — far
        better than arXiv's default OR behavior on space-separated terms.
        """
        try:
            text = complete(
                self.client,
                model=self._model,
                max_tokens=2000,  # headroom for thinking
                system=self._QUERY_SYSTEM,
                user=f"Topic: {topic}",
            ).strip()
            query = _sanitize_query(text)
            if query:
                return query
            logger.warning("arXiv query rewrite was empty; using fallback")
        except Exception:
            logger.exception("arXiv query rewrite failed; using fallback")
        return _and_join(topic)

    def _search_page(
        self, query: str, max_results: int, start: int = 0
    ) -> list[Paper]:
        logger.info(
            "Searching arXiv with %r (%d results from offset %d)",
            query,
            max_results,
            start,
        )
        try:
            return self._fetch_page(query, max_results, start)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            # arXiv rejected the query syntax (e.g. a malformed LLM rewrite).
            # Retry once with a defensively simplified AND-join of its terms.
            safe = _and_join(query)
            logger.warning(
                "arXiv rejected query %r (400); retrying with %r", query, safe
            )
            return self._fetch_page(safe, max_results, start)

    def _fetch_page(
        self, query: str, max_results: int, start: int
    ) -> list[Paper]:
        params = urllib.parse.urlencode(
            {
                "search_query": query,
                "start": start,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        url = f"{ARXIV_API_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            feed = resp.read()
        return self._parse_feed(feed)

    @classmethod
    def _parse_feed(cls, feed: bytes) -> list[Paper]:
        root = ET.fromstring(feed)
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", _ARXIV_NS):
            title = _text(entry, "atom:title")
            abstract = _text(entry, "atom:summary")
            entry_id = _text(entry, "atom:id")
            published = _text(entry, "atom:published")
            authors = [
                _text(a, "atom:name")
                for a in entry.findall("atom:author", _ARXIV_NS)
            ]
            pdf_url = ""
            for link in entry.findall("atom:link", _ARXIV_NS):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
                    break
            if not pdf_url and entry_id:
                # Fall back to deriving the PDF URL from the abstract page id.
                pdf_url = entry_id.replace("/abs/", "/pdf/")
            if not (title and pdf_url):
                continue
            papers.append(
                Paper(
                    paper_id=entry_id,
                    title=clean(title),
                    authors=[clean(a) for a in authors if a],
                    published=published,
                    abstract=clean(abstract),
                    pdf_url=pdf_url,
                    source=cls.name,
                )
            )
        logger.info("Found %d papers", len(papers))
        return papers


def _text(element: ET.Element, path: str) -> str:
    found = element.find(path, _ARXIV_NS)
    return found.text or "" if found is not None else ""


# Words that rarely appear in scholarly metadata; dropped from the fallback
# query so they don't over-constrain an AND search (see the 'sota' analysis).
_STOPWORDS = {
    "sota", "state-of-the-art", "state", "of", "the", "art", "best", "novel",
    "latest", "using", "use", "approach", "method", "methods", "paper",
    "papers", "research", "study", "a", "an", "for", "to", "and", "on", "with",
}


# arXiv query syntax tokens to ignore when deriving a fallback from a string
# that may already contain query syntax (e.g. a rejected LLM rewrite).
_SYNTAX_TOKENS = {"all", "ti", "abs", "au", "cat", "co", "and", "or", "andnot"}


def _and_join(topic: str) -> str:
    """Fallback query: AND-join the meaningful words as ``all:`` terms.

    Strips field prefixes, boolean operators, and light stopwords so the AND
    search isn't over-constrained (and so it's safe to call on a string that
    already contains arXiv syntax). No LLM. Always far better than arXiv's
    OR-on-spaces default. If nothing survives filtering, uses the raw words.
    """
    # Drop "field:" prefixes so we keep the term, not the field name.
    cleaned = re.sub(r"\b(?:all|ti|abs|au|cat|co)\s*:", " ", topic)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-]*", cleaned)
    terms = [
        w
        for w in words
        if w.lower() not in _STOPWORDS and w.lower() not in _SYNTAX_TOKENS
    ]
    if not terms:
        terms = words or [topic]
    # Dedupe while preserving order.
    seen: set[str] = set()
    unique = [t for t in terms if not (t.lower() in seen or seen.add(t.lower()))]
    return " AND ".join(f"all:{w}" for w in unique)


def _sanitize_query(text: str) -> str:
    """Validate/clean an LLM-produced arXiv query string.

    Guards against the model wrapping the query in prose or code fences by
    extracting the substring from the first field prefix to the end of that
    line. Returns "" if no field prefix is present (so the caller falls back).
    """
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip()
        if not line:
            continue
        # A field prefix must be present somewhere for this to be a query line.
        m = re.search(r"\b(?:all|ti|abs|au|cat|co)\s*:", line)
        if not m:
            continue
        # Drop leading prose ("Here is your query: ...") only when the query
        # doesn't legitimately open with '(' or a field prefix — otherwise we'd
        # strip a real leading paren and unbalance the query.
        if not re.match(r"^\(*\s*(?:all|ti|abs|au|cat|co)\s*:", line):
            line = line[m.start():].strip()
        # Reject malformed queries (e.g. unbalanced parens/quotes) so the caller
        # falls back rather than sending arXiv a 400.
        if line.count("(") != line.count(")"):
            return ""
        if line.count('"') % 2 != 0:
            return ""
        return line
    return ""
