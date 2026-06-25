from .base import ParsedContent, Parser, SectionContent
from .html_parser import HtmlParser
from .pdf_parser import PdfParser
from .plaintext_parser import PlaintextParser

__all__ = [
    "ParsedContent",
    "Parser",
    "SectionContent",
    "HtmlParser",
    "PdfParser",
    "PlaintextParser",
    "get_parsers",
]


def get_parsers(*, pdf_max_pages: int = 100) -> list:
    """Return parsers in priority order. PlaintextParser is always last (catch-all)."""
    return [HtmlParser(), PdfParser(max_pages=pdf_max_pages), PlaintextParser()]
