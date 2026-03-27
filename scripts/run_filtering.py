import json
from datetime import datetime
from bioscancast.filtering.pipeline import FilteringPipeline
from bioscancast.filtering.models import ForecastQuestion, SearchResult


def serialize(doc):
    return {
        "result_id": doc.result_id,
        "url": doc.url,
        "domain": doc.domain,
        "title": doc.title,
        "final_score": doc.final_score,
        "relevance_score": doc.relevance_score,
        "credibility_score": doc.credibility_score,
        "selection_reasons": doc.selection_reasons,
        "extraction_priority": doc.extraction_priority,
        "extraction_mode": doc.extraction_mode,
        "expected_value": doc.expected_value,
    }


def main():
    question = ForecastQuestion(
        id="q1",
        text="Will country X report more than 50 cases of pathogen Y?",
        created_at=datetime.now(),
    )

    search_results = [
        SearchResult(
            id="r1",
            question_id="q1",
            query_id="sq1",
            engine="google",
            url="https://who.int/update",
            canonical_url="https://who.int/update",
            domain="who.int",
            title="Outbreak update pathogen Y country X",
            snippet="Confirmed human cases reported",
            rank=1,
            retrieved_at=datetime.now(),
            source_tier="official",
            is_official_domain=True,
            domain_score=1.0,
            freshness_score=0.9,
        )
    ]

    pipeline = FilteringPipeline()
    docs = pipeline.run(question, search_results)

    output = [serialize(d) for d in docs]

    with open("data/filtered_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()