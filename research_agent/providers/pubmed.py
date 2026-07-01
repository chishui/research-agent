"""PubMed paper provider (abstract-based, via NCBI E-utilities).

PubMed indexes biomedical and life-sciences literature (MEDLINE). Unlike arXiv
there is no open PDF — E-utilities returns metadata and the abstract, while full
text lives behind PMC or publisher paywalls. So this is an *abstract provider*:
``_search_page`` runs ``esearch`` (topic -> PMIDs) then ``efetch`` (PMIDs ->
metadata + abstract) in one shot and stashes the abstract on each ``Paper``;
``_fetch`` just packages that into a ``ParsedPaper`` (no second network call,
no PDF download).

API: NCBI E-utilities — documented and stable.
  https://www.ncbi.nlm.nih.gov/books/NBK25500/

Note on rate limits: NCBI allows ~3 requests/sec without an API key. We make 2
requests per page (esearch + efetch), so default usage stays well under that.
Set ``NCBI_API_KEY`` in the environment to raise the limit if needed.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

from common import Paper, ParsedPaper

from .base import PaperProvider, clean

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_USER_AGENT = "agent-research-tool/1.0 (+https://example.com)"


class PubMedProvider(PaperProvider):
    """Searches PubMed and parses abstracts via NCBI E-utilities."""

    name = "pubmed"
    description = (
        "Searches PubMed (MEDLINE), the U.S. National Library of Medicine's "
        "database of biomedical and life-sciences literature, covering "
        "peer-reviewed clinical and biological research."
    )
    realms = (
        "medicine",
        "clinical research",
        "nutrition",
        "biology",
        "genetics",
        "neuroscience",
        "pharmacology",
        "public health",
        "epidemiology",
    )

    # ----------------------------------------------------------------- #
    # Stage 1: one page of results -> Paper metadata (with abstracts)   #
    # ----------------------------------------------------------------- #
    def _search_page(
        self, query: str, max_results: int, start: int = 0
    ) -> list[Paper]:
        pmids = self._esearch(query, max_results, start)
        if not pmids:
            return []
        return self._efetch(pmids)

    def _esearch(self, query: str, retmax: int, retstart: int) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retstart": retstart,
            "sort": "relevance",
            "retmode": "json",
        }
        self._add_api_key(params)
        url = f"{EUTILS_BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
        logger.info(
            "Searching PubMed for %r (%d results from offset %d)",
            query,
            retmax,
            retstart,
        )
        data = json.loads(self._get(url))
        result = data.get("esearchresult", {})
        return result.get("idlist", [])

    def _efetch(self, pmids: list[str]) -> list[Paper]:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        self._add_api_key(params)
        url = f"{EUTILS_BASE}/efetch.fcgi?{urllib.parse.urlencode(params)}"
        xml = self._get(url)
        papers = self._parse_articles(xml)
        logger.info("Fetched %d PubMed articles", len(papers))
        return papers

    @classmethod
    def _parse_articles(cls, xml: bytes) -> list[Paper]:
        root = ET.fromstring(xml)
        papers: list[Paper] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            title_el = art.find(".//ArticleTitle")
            title = clean("".join(title_el.itertext())) if title_el is not None else ""

            # Abstracts may have multiple labeled sections; join them.
            abstract = clean(
                " ".join(
                    "".join(node.itertext())
                    for node in art.findall(".//Abstract/AbstractText")
                )
            )

            authors = []
            for a in art.findall(".//AuthorList/Author"):
                last = a.findtext("LastName")
                fore = a.findtext("ForeName") or a.findtext("Initials")
                if last:
                    authors.append(clean(f"{fore or ''} {last}".strip()))

            published = (
                art.findtext(".//PubDate/Year")
                or art.findtext(".//PubDate/MedlineDate")
                or ""
            )

            if not (pmid and title):
                continue
            papers.append(
                Paper(
                    paper_id=f"PMID:{pmid}",
                    title=title,
                    authors=authors,
                    published=published,
                    abstract=abstract,
                    # No open PDF; link to the PubMed landing page.
                    pdf_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    source=cls.name,
                )
            )
        return papers

    # ----------------------------------------------------------------- #
    # Fetch: abstract already captured during search, just package it   #
    # ----------------------------------------------------------------- #
    def _fetch(self, paper: Paper) -> ParsedPaper:
        # PubMed gives us no open full text, so introduction/conclusion are
        # empty — the abstract (captured during _search_page) is enough for
        # relevance + summarization.
        return ParsedPaper(
            paper=paper,
            abstract=paper.abstract,
            introduction="",
            conclusion="",
        )

    # ----------------------------------------------------------------- #
    # Helpers                                                           #
    # ----------------------------------------------------------------- #
    @staticmethod
    def _add_api_key(params: dict) -> None:
        api_key = os.environ.get("NCBI_API_KEY")
        if api_key:
            params["api_key"] = api_key

    def _get(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return resp.read()
