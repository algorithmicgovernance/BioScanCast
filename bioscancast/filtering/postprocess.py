from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .models import FilterDecision, FilteredDocument, SearchResult


def build_filtered_documents(
    decisions: List[FilterDecision],
    result_map: Dict[str, SearchResult],
) -> List[FilteredDocument]:
    docs: list[FilteredDocument] = []

    for decision in decisions:
        if decision.keep is not True:
            continue

        result = result_map[decision.result_id]
        docs.append(
            FilteredDocument(
                result_id=result.id,
                question_id=result.question_id,
                url=result.url,
                canonical_url=result.canonical_url,
                domain=result.domain,
                title=result.title,
                snippet=result.snippet,
                published_date=result.published_date,
                file_type=result.file_type,
                relevance_score=decision.relevance_score,
                credibility_score=decision.credibility_score,
                final_score=decision.priority_score,
                source_tier=result.source_tier,
                is_official_domain=result.is_official_domain,
                selection_reasons=decision.reason_codes,
                extraction_priority=0,
                extraction_mode="unknown",
                expected_value="medium",
            )
        )

    docs.sort(key=lambda d: d.final_score, reverse=True)
    return docs


def cap_per_domain_and_type(
    docs: List[FilteredDocument],
    max_docs_per_domain: int,
    max_docs_per_type: int,
) -> List[FilteredDocument]:
    kept: list[FilteredDocument] = []
    domain_counts = defaultdict(int)
    type_counts = defaultdict(int)

    for doc in docs:
        doc_type = doc.file_type or "unknown"

        if domain_counts[doc.domain] >= max_docs_per_domain:
            continue
        if type_counts[doc_type] >= max_docs_per_type:
            continue

        kept.append(doc)
        domain_counts[doc.domain] += 1
        type_counts[doc_type] += 1

    return kept


def assign_extraction_hints(docs: List[FilteredDocument]) -> List[FilteredDocument]:
    for idx, doc in enumerate(docs, start=1):
        doc.extraction_priority = idx

        if (doc.file_type or "").lower() == "pdf":
            doc.extraction_mode = "pdf"
        elif doc.url.lower().endswith(".pdf"):
            doc.extraction_mode = "pdf"
        elif doc.url.lower().startswith("http"):
            doc.extraction_mode = "html"
        else:
            doc.extraction_mode = "unknown"

        if doc.is_official_domain and doc.final_score >= 0.85:
            doc.expected_value = "high"
        elif doc.source_tier in {"official", "academic"}:
            doc.expected_value = "medium"
        else:
            doc.expected_value = "low"

    return docs