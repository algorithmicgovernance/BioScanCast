from __future__ import annotations

import json as _json
import logging
import re
from datetime import datetime
from typing import Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup, Tag

from .base import ParsedContent, SectionContent

try:
    import trafilatura
except ImportError:
    trafilatura = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Minimum extractable content for the trafilatura-XML path to be accepted.
# When the XML output's body text is shorter than this, we treat the
# extraction as having failed (e.g. listing pages where trafilatura can
# only isolate a snippet) and fall back to the DOM walker. The threshold
# is intentionally small so single-paragraph articles still go down the
# preferred path.
_MIN_TRAFILATURA_BODY_CHARS = 200


class HtmlParser:
    """Extracts structured content from HTML documents.

    Strategy: prefer trafilatura's structured (XML) main-content extraction
    which strips navigation, sidebars, "related articles" sections, and
    other site boilerplate that a raw DOM walk would otherwise pull in.
    Fall back to a DOM walker when trafilatura returns too little
    content — usually because the page is a listing/landing page
    trafilatura can't isolate, or because trafilatura is not installed.
    """

    def can_parse(self, content_type: str, content: bytes) -> bool:
        if "html" in (content_type or ""):
            return True
        head = content[:128].lstrip().lower()
        return head.startswith((b"<!doctype", b"<html"))

    def parse(self, content: bytes, *, source_url: str) -> ParsedContent:
        html_text = content.decode("utf-8", errors="replace")

        # Use trafilatura for cleaned main-content text (plain) AND
        # the structural XML output used for section extraction below.
        main_text = ""
        main_xml = ""
        if trafilatura is not None:
            main_text = trafilatura.extract(html_text) or ""
            main_xml = (
                trafilatura.extract(
                    html_text,
                    output_format="xml",
                    include_tables=True,
                    include_links=False,
                )
                or ""
            )

        # Parse with BeautifulSoup for document-level metadata (title,
        # date, language) — these are reliably in the raw DOM head/meta
        # tags whether or not the body extraction succeeds.
        soup = BeautifulSoup(html_text, "html.parser")
        title = self._extract_title(soup)
        published_date = self._extract_published_date(soup, source_url)
        language = self._extract_language(soup)

        # Primary path: walk trafilatura's main-content XML.
        sections = (
            self._extract_sections_from_trafilatura_xml(main_xml)
            if main_xml
            else []
        )
        # Fallback path: walk the full DOM when trafilatura's output is
        # too thin to be trustworthy (listing pages, error pages,
        # pages trafilatura's heuristics misjudge).
        if not sections:
            logger.debug(
                "trafilatura XML extraction yielded no usable sections for %s "
                "(xml_chars=%d); falling back to DOM walker",
                source_url, len(main_xml),
            )
            sections = self._extract_sections(soup)

        raw_text = main_text or soup.get_text(separator="\n", strip=True)

        return ParsedContent(
            raw_text=raw_text,
            sections=sections if sections else self._fallback_sections(main_text),
            title=title,
            language=language,
            published_date=published_date,
        )

    # ----------------------------------------------------------------
    # Trafilatura XML → sections
    # ----------------------------------------------------------------

    def _extract_sections_from_trafilatura_xml(
        self, xml_text: str
    ) -> List[SectionContent]:
        """Walk trafilatura's XML output (``<doc><main>...</main></doc>``)
        to produce ordered sections.

        Trafilatura emits headings as ``<head rend="hN">``, paragraphs as
        ``<p>``, and tables as ``<table>`` with ``<row>`` containing
        ``<cell>`` (sometimes wrapping a ``<p>``). We rebuild a
        heading-stack-aware section list with the same shape as the DOM
        walker so downstream code doesn't care which path produced them.

        Returns an empty list when the XML body has less than
        ``_MIN_TRAFILATURA_BODY_CHARS`` of body text — the caller treats
        that as a signal to fall back to the DOM walker.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("Could not parse trafilatura XML: %s", exc)
            return []

        main_el = root.find("main")
        if main_el is None:
            return []

        # Cheap quality gate: count printable body text. Headings + para
        # text + cell text together must clear the threshold or we drop
        # to the DOM walker.
        body_chars = sum(
            len(_collect_element_text(el))
            for el in main_el.iter()
            if el.tag in ("head", "p", "cell")
        )
        if body_chars < _MIN_TRAFILATURA_BODY_CHARS:
            return []

        sections: List[SectionContent] = []
        heading_stack: List[str] = []
        current_level = 0
        current_text_parts: List[str] = []

        def flush_prose() -> None:
            if not current_text_parts:
                return
            text = "\n".join(current_text_parts).strip()
            if text:
                sections.append(
                    SectionContent(
                        section_path=" > ".join(heading_stack) if heading_stack else None,
                        page_number=None,
                        text=text,
                        chunk_type="prose",
                        extractor="trafilatura",
                    )
                )
            current_text_parts.clear()

        for child in main_el:
            tag = child.tag
            if tag == "head":
                flush_prose()
                level = _heading_level(child.get("rend"))
                heading_text = _collect_element_text(child)
                if level <= current_level:
                    heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading_text)
                current_level = level

            elif tag == "p" or tag == "quote":
                p_text = _collect_element_text(child).strip()
                if p_text:
                    current_text_parts.append(p_text)

            elif tag == "list":
                # Render a list as one prose chunk with bullet-marked lines.
                items = [
                    f"• {_collect_element_text(item).strip()}"
                    for item in child.findall("item")
                    if _collect_element_text(item).strip()
                ]
                if items:
                    current_text_parts.append("\n".join(items))

            elif tag == "table":
                flush_prose()
                table_rows = _parse_trafilatura_table(child)
                if table_rows:
                    sections.append(
                        SectionContent(
                            section_path=" > ".join(heading_stack) if heading_stack else None,
                            page_number=None,
                            text="",
                            chunk_type="table",
                            table_rows=table_rows,
                            extractor="trafilatura",
                        )
                    )

        flush_prose()
        return sections

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):  # type: ignore[union-attr]
            return og_title["content"].strip()  # type: ignore[index]
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return None

    def _extract_published_date(
        self, soup: BeautifulSoup, source_url: str = ""
    ) -> Optional[datetime]:
        """Extract a publication date from HTML metadata or URL structure.

        Tries multiple conventions in priority order (publication-semantic
        metadata first, then URL path patterns, then generic time tags,
        then modification-semantic as a last resort), and parses the
        first candidate that yields a valid datetime. Returns ``None``
        when no usable date is found — legitimately the case for listing
        pages or org-site landings that don't expose one. Body-text date
        extraction is intentionally NOT attempted here: too many pages
        (especially epidemiology articles) contain dozens of unrelated
        dates that a regex would mistakenly pick up.

        Why ``DC.date`` without an explicit "issued" / "created" suffix
        is NOT consulted: in practice (e.g. CDC HAN alerts) it's used as
        a last-rendered timestamp, not a publication date, and returning
        a misleading post-publication date is worse than returning None.
        """
        for label, raw in self._iter_date_candidates(soup, source_url):
            dt = self._parse_date(raw)
            if dt is not None:
                logger.debug("pub_date matched on %s: %r", label, raw)
                return dt
        return None

    def _iter_date_candidates(
        self, soup: BeautifulSoup, source_url: str = "",
    ) -> Iterable[Tuple[str, str]]:
        """Yield ``(label, raw_string)`` candidates in priority order.

        The label is for diagnostic logging only; the order is what
        actually matters. Publication-semantic sources come first
        (``article:published_time``, JSON-LD ``datePublished``, Dublin
        Core ``issued``/``created``). Generic ``<time datetime>`` lands
        in the middle. Modification-semantic sources come last so they
        only contribute when nothing better is available.
        """
        # ---- 1. Publication-semantic <meta> ----
        for prop in (
            "article:published_time",
            "og:article:published_time",
            "og:published_time",
        ):
            m = soup.find("meta", property=prop)
            if m and m.get("content"):
                yield f"meta[property={prop}]", str(m["content"])
        for name in (
            "sailthru.date",
            "parsely-pub-date",
            "article.published",
            "pubdate",
        ):
            m = soup.find("meta", attrs={"name": name})
            if m and m.get("content"):
                yield f"meta[name={name}]", str(m["content"])

        # ---- 2. JSON-LD datePublished ----
        for value in self._iter_jsonld_date_values(soup, "datePublished"):
            yield "jsonld[datePublished]", value

        # ---- 3. Dublin Core publication-semantic ----
        for name in (
            "DC.date.issued",
            "dcterms:issued",
            "DC.date.created",
            "dcterms:created",
        ):
            m = soup.find("meta", attrs={"name": name})
            if m and m.get("content"):
                yield f"meta[name={name}]", str(m["content"])

        # ---- 4. Legacy / generic publication_date ----
        m = soup.find("meta", attrs={"name": "publication_date"})
        if m and m.get("content"):
            yield "meta[name=publication_date]", str(m["content"])

        # ---- 5. JSON-LD dateCreated ----
        for value in self._iter_jsonld_date_values(soup, "dateCreated"):
            yield "jsonld[dateCreated]", value

        # ---- 6. URL path date pattern ----
        # Lower priority than any publication-semantic metadata above,
        # higher than the generic <time> tag below — URL dates are
        # structurally bounded (no false positives from arbitrary body
        # text) but often only give year precision, while <time> tags
        # may be more precise but can refer to unrelated timestamps
        # (sidebar timestamps, last-modified ribbons, related articles).
        # Stronger semantic grounding wins over potentially-finer
        # precision from an ambiguous source.
        url_iso = _extract_url_date_iso(source_url)
        if url_iso is not None:
            yield "url[path]", url_iso

        # ---- 7. <time datetime=...> (first one, intentionally) ----
        t = soup.find("time", attrs={"datetime": True})
        if t:
            yield "time[datetime]", str(t["datetime"])

        # ---- 8. Modification-semantic (last resort) ----
        for prop in ("article:modified_time", "og:modified_time"):
            m = soup.find("meta", property=prop)
            if m and m.get("content"):
                yield f"meta[property={prop}]", str(m["content"])
        for value in self._iter_jsonld_date_values(soup, "dateModified"):
            yield "jsonld[dateModified]", value

    def _iter_jsonld_date_values(
        self, soup: BeautifulSoup, key: str
    ) -> Iterable[str]:
        """Walk every ``<script type="application/ld+json">`` block in
        the document and yield string values whose key matches ``key``
        at any depth. JSON-LD nests freely (e.g. an Article inside a
        NewsArticle inside a WebPage); a flat ``obj.get(key)`` would
        miss the common cases.

        Malformed JSON is silently skipped — never raised — because
        partial / templated JSON-LD blocks are common in the wild.
        """
        for script in soup.find_all("script", type="application/ld+json"):
            text = script.string or ""
            if not text.strip():
                continue
            try:
                obj = _json.loads(text)
            except (_json.JSONDecodeError, TypeError):
                continue
            yield from _walk_jsonld_for_key(obj, key)

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        date_str = (date_str or "").strip()
        if not date_str:
            return None
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def _extract_language(self, soup: BeautifulSoup) -> Optional[str]:
        html_tag = soup.find("html")
        if html_tag and isinstance(html_tag, Tag):
            lang = html_tag.get("lang")
            if lang:
                return str(lang).split("-")[0].lower()
        return None

    def _extract_sections(self, soup: BeautifulSoup) -> List[SectionContent]:
        """Walk the DOM to extract heading-delimited sections and tables."""
        body = soup.find("body") or soup
        sections: List[SectionContent] = []
        heading_stack: List[str] = []  # tracks h1..h4 nesting
        current_level = 0
        current_text_parts: List[str] = []

        heading_tags = {"h1", "h2", "h3", "h4"}

        for element in body.descendants:
            if not isinstance(element, Tag):
                continue

            tag_name = element.name

            if tag_name in heading_tags:
                # Flush accumulated text before this heading
                if current_text_parts:
                    text = "\n".join(current_text_parts).strip()
                    if text:
                        sections.append(
                            SectionContent(
                                section_path=" > ".join(heading_stack) if heading_stack else None,
                                page_number=None,
                                text=text,
                                chunk_type="prose",
                            )
                        )
                    current_text_parts = []

                # Update heading stack
                level = int(tag_name[1])
                heading_text = element.get_text(strip=True)
                if level <= current_level:
                    heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading_text)
                current_level = level

            elif tag_name == "table":
                # Flush text before table
                if current_text_parts:
                    text = "\n".join(current_text_parts).strip()
                    if text:
                        sections.append(
                            SectionContent(
                                section_path=" > ".join(heading_stack) if heading_stack else None,
                                page_number=None,
                                text=text,
                                chunk_type="prose",
                            )
                        )
                    current_text_parts = []

                table_rows = self._parse_table(element)
                if table_rows:
                    caption = element.find("caption")
                    caption_text = caption.get_text(strip=True) if caption else ""
                    sections.append(
                        SectionContent(
                            section_path=" > ".join(heading_stack) if heading_stack else None,
                            page_number=None,
                            text=caption_text,
                            chunk_type="table",
                            table_rows=table_rows,
                        )
                    )

            elif tag_name == "p":
                p_text = element.get_text(strip=True)
                if p_text:
                    current_text_parts.append(p_text)

        # Flush remaining text
        if current_text_parts:
            text = "\n".join(current_text_parts).strip()
            if text:
                sections.append(
                    SectionContent(
                        section_path=" > ".join(heading_stack) if heading_stack else None,
                        page_number=None,
                        text=text,
                        chunk_type="prose",
                    )
                )

        return sections

    def _parse_table(self, table_tag: Tag) -> List[List[str]]:
        rows: List[List[str]] = []
        for tr in table_tag.find_all("tr"):
            cells = []
            for td in tr.find_all(["th", "td"]):
                cells.append(td.get_text(strip=True))
            if cells:
                rows.append(cells)
        return rows

    def _fallback_sections(self, text: str) -> List[SectionContent]:
        """When structured extraction yields nothing, split on blank lines."""
        if not text:
            return []
        paragraphs = re.split(r"\n\s*\n", text)
        sections = []
        for para in paragraphs:
            para = para.strip()
            if para:
                sections.append(
                    SectionContent(
                        section_path=None,
                        page_number=None,
                        text=para,
                        chunk_type="prose",
                    )
                )
        return sections


# ---------------------------------------------------------------------------
# Module-level helpers for the trafilatura XML walker
# ---------------------------------------------------------------------------


def _collect_element_text(el: ET.Element) -> str:
    """Concatenate text from an XML element and all its descendants."""
    parts: List[str] = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_collect_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _heading_level(rend: Optional[str]) -> int:
    """Parse trafilatura's ``rend="hN"`` heading marker. Defaults to h2
    when the marker is missing or unrecognised (matches the most common
    article structure)."""
    if rend and rend.startswith("h") and rend[1:].isdigit():
        return max(1, min(int(rend[1:]), 4))
    return 2


def _parse_trafilatura_table(table_el: ET.Element) -> List[List[str]]:
    """Convert a trafilatura ``<table>`` element into row-major cells.

    Trafilatura emits ``<row>`` containing ``<cell>``, sometimes with a
    nested ``<p>``. We collapse to plain text per cell and discard any
    rows that came out empty.
    """
    rows: List[List[str]] = []
    for row in table_el.findall("row"):
        cells = [_collect_element_text(c).strip() for c in row.findall("cell")]
        if any(cells):
            rows.append(cells)
    return rows


def _walk_jsonld_for_key(obj, key: str) -> Iterable[str]:
    """Recursively yield string values whose key matches ``key`` anywhere
    in a JSON-LD object tree.

    JSON-LD documents nest deeply — a NewsArticle inside a WebPage
    inside an @graph list is common — so we can't just look at top-level
    keys. We yield only string values; date keys that point to nested
    objects (rare) are ignored.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, str):
                yield v
            yield from _walk_jsonld_for_key(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_jsonld_for_key(item, key)


# ---------------------------------------------------------------------------
# URL path date extraction
# ---------------------------------------------------------------------------

# Date patterns we recognise in URL path segments, in order of how
# specifically they pin down a date. The patterns are intentionally
# narrow: each must occupy a full path segment (between two ``/``) or
# anchored to the start of one, so arbitrary 4-digit numbers buried
# inside slugs ("section-2024-summary") do NOT count as years.
_URL_FULL_DATE_HYPHEN_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_URL_FULL_DATE_COMPACT_RE = re.compile(r"^(\d{8})$")  # YYYYMMDD
_URL_YEAR_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_URL_YEAR_ONLY_RE = re.compile(r"^(\d{4})$")
# Year-prefixed identifiers used by WHO ("2024-DON530"), CDC report
# numbers, and many news-site categorical IDs. The year must be followed
# by a separator (``-`` or ``_``) so plain integers like "20240813" go
# to the compact-date matcher instead.
_URL_YEAR_PREFIX_RE = re.compile(r"^(\d{4})[-_][A-Za-z].*$")

# Bounds on a plausible publication year. 1990 is a generous floor
# (article timestamps in our domain rarely predate the modern web);
# 2100 keeps us safe against typoed far-future years.
_URL_MIN_YEAR = 1990
_URL_MAX_YEAR = 2100


def _extract_url_date_iso(url: str) -> Optional[str]:
    """Try to extract a publication date from URL path segments.

    Returns an ISO-formatted date string (``YYYY-MM-DD``) suitable for
    feeding into ``_parse_date``, or ``None`` when no date-shaped
    structure is found. Year-only matches default to January 1;
    year-month matches default to the 1st of the month. Downstream
    consumers see a ``datetime`` either way.

    The matcher checks segments in this order of specificity, returning
    the first match found:

      1. Full ISO date in one segment: ``/2024-08-13/`` or ``/20240813/``
      2. Three consecutive segments: ``/2024/08/13/``
      3. Year-month in one segment: ``/2024-08/``
      4. Two consecutive segments: ``/2024/08/``
      5. Year-prefixed slug: ``/2024-DON530`` or ``/2024-q3-report``
      6. Bare year segment: ``/2024/``

    Patterns 5 and 6 only return year-precision dates. Patterns
    requiring separators around the year prevent false positives
    from numeric IDs (``/article/20240/``).
    """
    if not url:
        return None
    try:
        from urllib.parse import urlparse
        path = urlparse(url).path
    except Exception:
        return None
    segments = [s for s in path.split("/") if s]
    if not segments:
        return None

    # 1. Full ISO date in a single segment (most specific)
    for seg in segments:
        m = _URL_FULL_DATE_HYPHEN_RE.match(seg)
        if m and _valid_date(int(m.group(1)), int(m.group(2)), int(m.group(3))):
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        m = _URL_FULL_DATE_COMPACT_RE.match(seg)
        if m:
            ymd = m.group(1)
            y, mo, d = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
            if _valid_date(y, mo, d):
                return f"{y:04d}-{mo:02d}-{d:02d}"

    # 2. Three consecutive numeric segments: /YYYY/MM/DD/
    for i in range(len(segments) - 2):
        ys, ms, ds = segments[i], segments[i + 1], segments[i + 2]
        if _URL_YEAR_ONLY_RE.match(ys) and ms.isdigit() and ds.isdigit():
            y, mo, d = int(ys), int(ms), int(ds)
            if _valid_date(y, mo, d):
                return f"{y:04d}-{mo:02d}-{d:02d}"

    # 3. Year-month in one segment: /YYYY-MM/
    for seg in segments:
        m = _URL_YEAR_MONTH_RE.match(seg)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if _valid_year_month(y, mo):
                return f"{y:04d}-{mo:02d}-01"

    # 4. Two consecutive numeric segments: /YYYY/MM/
    for i in range(len(segments) - 1):
        ys, ms = segments[i], segments[i + 1]
        if _URL_YEAR_ONLY_RE.match(ys) and ms.isdigit():
            y, mo = int(ys), int(ms)
            if _valid_year_month(y, mo):
                return f"{y:04d}-{mo:02d}-01"

    # 5. Year-prefixed slug: /2024-DON530, /2024-q3-summary
    for seg in segments:
        m = _URL_YEAR_PREFIX_RE.match(seg)
        if m:
            y = int(m.group(1))
            if _URL_MIN_YEAR <= y <= _URL_MAX_YEAR:
                return f"{y:04d}-01-01"

    # 6. Bare year segment: /2024/
    for seg in segments:
        m = _URL_YEAR_ONLY_RE.match(seg)
        if m:
            y = int(m.group(1))
            if _URL_MIN_YEAR <= y <= _URL_MAX_YEAR:
                return f"{y:04d}-01-01"

    return None


def _valid_year_month(y: int, mo: int) -> bool:
    return _URL_MIN_YEAR <= y <= _URL_MAX_YEAR and 1 <= mo <= 12


def _valid_date(y: int, mo: int, d: int) -> bool:
    if not _valid_year_month(y, mo) or not (1 <= d <= 31):
        return False
    try:
        datetime(y, mo, d)
    except ValueError:
        return False
    return True
