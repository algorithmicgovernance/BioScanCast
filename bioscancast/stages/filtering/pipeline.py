from __future__ import annotations

from typing import List, Optional

from bioscancast.llm.base import LLMClient

from .config import FILTER_CONFIG
from .deduplication import deduplicate_filtered_documents
from .heuristics import heuristic_filter
from .llm_filter import llm_filter_candidates
from .models import FilterDecision, FilteredDocument, ForecastQuestion, SearchResult
from .postprocess import assign_extraction_hints, build_filtered_documents, cap_per_domain_and_type
from .reranker import rerank_borderline_candidates, split_for_llm_review


class FilteringPipeline:
    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client

    def run(
        self,
        question: ForecastQuestion,
        search_results: List[SearchResult],
    ) -> List[FilteredDocument]:
        result_map = {r.id: r for r in search_results}

        heuristic_keep, borderline, _rejected = heuristic_filter(search_results, question)

        reranked = rerank_borderline_candidates(
            borderline=borderline,
            result_map=result_map,
            question=question,
        )

        llm_needed, llm_not_needed = split_for_llm_review(reranked)

        llm_decisions: list[FilterDecision] = []
        if llm_needed:
            if self.llm_client is None:
                # No LLM client. Default is fail-closed (reject the ambiguous
                # band). When the soft-fallback flag is enabled, keep candidates
                # that are official-domain or sufficiently relevant — see
                # FILTER_CONFIG["no_llm_soft_fallback"] and issue #13.
                soft = FILTER_CONFIG.get("no_llm_soft_fallback", False)
                rel_threshold = FILTER_CONFIG.get(
                    "no_llm_fallback_relevance_threshold", 0.5
                )
                for d in llm_needed:
                    d.stage = "llm_skipped"
                    result = result_map.get(d.result_id)
                    is_official = bool(result and result.is_official_domain)
                    if soft and (is_official or d.relevance_score >= rel_threshold):
                        d.keep = True
                        d.reason_codes.append("no_llm_soft_fallback_kept")
                    else:
                        d.keep = False
                        d.reason_codes.append("no_llm_client_configured")
                llm_decisions = llm_needed
            else:
                llm_decisions = llm_filter_candidates(
                    question=question,
                    candidate_decisions=llm_needed,
                    result_map=result_map,
                    llm_client=self.llm_client,
                )

        combined = heuristic_keep + llm_not_needed + llm_decisions

        docs = build_filtered_documents(combined, result_map)
        for doc in docs:
            doc.region = question.region
            doc.question_text = question.text
        docs = deduplicate_filtered_documents(docs)
        docs = cap_per_domain_and_type(
            docs,
            max_docs_per_domain=FILTER_CONFIG["max_docs_per_domain"],
            max_docs_per_type=FILTER_CONFIG["max_docs_per_type"],
        )
        docs = assign_extraction_hints(docs)

        return docs
