from __future__ import annotations

from typing import List

from .config import FILTER_CONFIG
from .models import FilteredDocument
from .utils import jaccard_similarity


def deduplicate_filtered_documents(docs: List[FilteredDocument]) -> List[FilteredDocument]:
    if not docs:
        return docs

    docs = sorted(docs, key=lambda d: d.final_score, reverse=True)
    kept: list[FilteredDocument] = []

    for doc in docs:
        is_duplicate = False

        for existing in kept:
            same_url = (
                doc.canonical_url is not None
                and existing.canonical_url is not None
                and doc.canonical_url == existing.canonical_url
            )
            near_duplicate = (
                doc.domain == existing.domain
                and jaccard_similarity(
                    f"{doc.title} {doc.snippet}",
                    f"{existing.title} {existing.snippet}",
                ) >= FILTER_CONFIG["near_duplicate_similarity_threshold"]
            )

            if same_url or near_duplicate:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(doc)

    return kept