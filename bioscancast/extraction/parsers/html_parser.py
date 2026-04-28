from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional

from bs4 import BeautifulSoup, Tag

from .base import ParsedContent, SectionContent

try:
    import trafilatura
except ImportError:
    trafilatura = None  # type: ignore[assignment]


class HtmlParser:
    """Extracts structured content from HTML documents."""

    def can_parse(self, content_type: str, content: bytes) -> bool:
        if "html" in (content_type or ""):
            return True
        head = content[:128].lstrip().lower()
        return head.startswith((b"<!doctype", b"<html"))

    def parse(self, content: bytes, *, source_url: str) -> ParsedContent:
        html_text = content.decode("utf-8", errors="replace")

        # Use trafilatura for cleaned main-content text
        main_text = ""
        if trafilatura is not None:
            main_text = trafilatura.extract(html_text) or ""

        # Parse with BeautifulSoup for structure
        soup = BeautifulSoup(html_text, "html.parser")

        title = self._extract_title(soup)
        published_date = self._extract_published_date(soup)
        language = self._extract_language(soup)
        sections = self._extract_sections(soup)

        raw_text = main_text or soup.get_text(separator="\n", strip=True)

        return ParsedContent(
            raw_text=raw_text,
            sections=sections if sections else self._fallback_sections(main_text),
            title=title,
            language=language,
            published_date=published_date,
        )

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

    def _extract_published_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        for attr in ("article:published_time", "og:published_time"):
            meta = soup.find("meta", property=attr)
            if meta and meta.get("content"):  # type: ignore[union-attr]
                return self._parse_date(meta["content"])  # type: ignore[index]

        meta_name = soup.find("meta", attrs={"name": "publication_date"})
        if meta_name and meta_name.get("content"):  # type: ignore[union-attr]
            return self._parse_date(meta_name["content"])  # type: ignore[index]

        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            return self._parse_date(time_tag["datetime"])  # type: ignore[index]

        return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        date_str = date_str.strip()
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
