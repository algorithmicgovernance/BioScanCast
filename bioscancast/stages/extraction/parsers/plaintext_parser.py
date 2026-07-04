from __future__ import annotations

import re
from typing import List

from .base import ParsedContent, SectionContent


class PlaintextParser:
    """Fallback parser for text/plain and unknown content types."""

    def can_parse(self, content_type: str, content: bytes) -> bool:
        if "text/plain" in (content_type or ""):
            return True
        # Accept as fallback for unknown types
        return True

    def parse(self, content: bytes, *, source_url: str) -> ParsedContent:
        text = content.decode("utf-8", errors="replace")
        paragraphs = re.split(r"\n\s*\n", text)

        sections: List[SectionContent] = []
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

        return ParsedContent(
            raw_text=text,
            sections=sections,
        )
