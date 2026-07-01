"""JAMA Network paper provider (HTML-based).

JAMA Network (https://jamanetwork.com) publishes JAMA and its specialty
journals. Unlike arXiv there is no open PDF/API: articles are HTML pages and
most full text is paywalled. So this provider is an *HTML provider* — it scrapes
the search results page for article links, then reads each article's HTML
landing page (no PDF download).

Section extraction relies on the Highwire ``citation_*`` ``<meta>`` tags that
JAMA (and most publishers) embed in article pages:

    <meta name="citation_title"    content="..."/>
    <meta name="citation_author"   content="..."/>      (repeated)
    <meta name="citation_date"     content="..."/>
    <meta name="citation_abstract" content="..."/>

Full text is usually paywalled, so we populate the abstract from the metadata
and leave introduction/conclusion empty when they aren't freely available.

⚠️ Live note: JAMA sits behind a Cloudflare JS challenge that returns HTTP 403
to plain HTTP clients (urllib/requests/httpx). Fetching therefore requires a
browser-capable ``HtmlClient`` (e.g. one backed by a headless browser) injected
via ``html_client=``. The parsing logic below is exercised against a fixture in
tests; the selectors are based on JAMA's published page structure and should be
re-verified against live HTML once a working fetcher is available.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from common import HtmlClient, Paper, ParsedPaper

from .base import PaperProvider, clean

logger = logging.getLogger(__name__)

JAMA_BASE_URL = "https://jamanetwork.com"
JAMA_SEARCH_URL = "https://jamanetwork.com/searchresults"
# SearchSourceType=24 scopes results to research articles (from the example URL).
_SEARCH_SOURCE_TYPE = 24
# Results per page is fixed by the site; we paginate via the `page` param.
_RESULTS_PER_PAGE = 10


class JamaProvider(PaperProvider):
    """Searches JAMA Network and parses article landing pages."""

    name = "jama"
    # Disabled by default: JAMA is behind a Cloudflare JS challenge that returns
    # 403 to plain HTTP clients, so it can't be reached from this environment.
    # PubMed covers the same biomedical realm and works. Re-enable by injecting
    # a browser-capable HtmlClient (see the module docstring).
    enabled = False
    description = (
        "Searches JAMA Network (the Journal of the American Medical "
        "Association and its specialty journals) for peer-reviewed clinical "
        "and biomedical research."
    )
    realms = (
        "medicine",
        "clinical research",
        "public health",
        "epidemiology",
        "biomedical science",
        "health policy",
    )

    def __init__(self, *args, html_client: HtmlClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._http = html_client or HtmlClient(timeout=self._timeout)

    # ----------------------------------------------------------------- #
    # Stage 1: one page of search results -> Paper metadata             #
    # ----------------------------------------------------------------- #
    def _search_page(
        self, topic: str, max_results: int, start: int = 0
    ) -> list[Paper]:
        # JAMA paginates by 1-based page number, not by record offset.
        page = (start // _RESULTS_PER_PAGE) + 1
        params = {
            "q": topic,
            "SearchSourceType": _SEARCH_SOURCE_TYPE,
            "page": page,
        }
        logger.info("Searching JAMA for %r (page %d)", topic, page)
        html = self._http.get(JAMA_SEARCH_URL, params=params)
        return self._parse_search_results(html)[:max_results]

    @classmethod
    def _parse_search_results(cls, html: str) -> list[Paper]:
        soup = BeautifulSoup(html, "html.parser")
        papers: list[Paper] = []
        seen: set[str] = set()
        for anchor in soup.select("a[href*='/fullarticle/'], a[href*='/article-abstract/']"):
            href = anchor.get("href", "")
            article_url = urljoin(JAMA_BASE_URL, href)
            article_id = _article_id(article_url)
            if not article_id or article_id in seen:
                continue
            seen.add(article_id)
            title = clean(anchor.get_text())
            papers.append(
                Paper(
                    paper_id=article_url,
                    title=title,
                    authors=[],
                    published="",
                    abstract="",  # filled in during _fetch from the article page
                    pdf_url=article_url,  # the HTML landing page (no PDF)
                    source=cls.name,
                )
            )
        logger.info("Found %d JAMA articles", len(papers))
        return papers

    # ----------------------------------------------------------------- #
    # Fetch + parse a single article (HTML, not PDF)                    #
    # ----------------------------------------------------------------- #
    def _fetch(self, paper: Paper) -> ParsedPaper:
        logger.info("Fetching JAMA article: %s", paper.pdf_url)
        html = self._http.get(paper.pdf_url)
        return self._parse_article(paper, html)

    @staticmethod
    def _parse_article(paper: Paper, html: str) -> ParsedPaper:
        soup = BeautifulSoup(html, "html.parser")

        def metas(name: str) -> list[str]:
            return [
                clean(m["content"])
                for m in soup.find_all("meta", attrs={"name": name})
                if m.get("content")
            ]

        def meta(name: str) -> str:
            vals = metas(name)
            return vals[0] if vals else ""

        # Enrich the metadata captured from search with the article page.
        title = meta("citation_title") or paper.title
        authors = metas("citation_author") or paper.authors
        published = (
            meta("citation_date")
            or meta("citation_publication_date")
            or paper.published
        )
        abstract = meta("citation_abstract") or paper.abstract

        # Fall back to an on-page abstract block if the meta tag is absent.
        if not abstract:
            block = soup.select_one(
                "div.abstract, section.abstract, div[class*='abstract']"
            )
            if block:
                abstract = clean(block.get_text(" "))

        paper.title = title
        paper.authors = authors
        paper.published = published
        paper.abstract = abstract

        # Full text is typically paywalled; intro/conclusion are left empty when
        # not freely available. The abstract alone is enough for relevance +
        # summarization.
        return ParsedPaper(
            paper=paper,
            abstract=abstract,
            introduction="",
            conclusion="",
        )


def _article_id(url: str) -> str:
    """Extract the trailing numeric article id from a JAMA article URL."""
    m = re.search(r"/(\d+)(?:[/?#]|$)", url)
    return m.group(1) if m else ""
