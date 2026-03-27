from __future__ import annotations

from typing import List, Optional

from .config import FILTER_CONFIG
from .deduplication import deduplicate_filtered_documents
from .heuristics import heuristic_filter
from .llm_filter import LLMClient, llm_filter_candidates
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
                # Fail closed: reject ambiguous cases if no LLM client is configured.
                for d in llm_needed:
                    d.keep = False
                    d.stage = "llm_skipped"
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
        docs = deduplicate_filtered_documents(docs)
        docs = cap_per_domain_and_type(
            docs,
            max_docs_per_domain=FILTER_CONFIG["max_docs_per_domain"],
            max_docs_per_type=FILTER_CONFIG["max_docs_per_type"],
        )
        docs = assign_extraction_hints(docs)

        return docs
