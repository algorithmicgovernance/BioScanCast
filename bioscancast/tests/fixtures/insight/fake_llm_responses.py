"""Pre-built LLMResponse objects for insight stage tests.

These responses are organized by document/chunk and designed to be
enqueued into a FakeLLMClient for deterministic testing.
"""

from bioscancast.llm.base import LLMResponse


# ---- WHO Sudan virus document responses ----

# Response for chunk-who-sudan-001-p1 (prose with case count)
SUDAN_PROSE_RESPONSE = LLMResponse(
    content={
        "facts": [
            {
                "event_type": "case_count",
                "confidence": 0.92,
                "location": "Mubende district, Uganda",
                "pathogen": "Sudan virus",
                "metric_name": "confirmed_cases",
                "metric_value": 9,
                "metric_unit": "cases",
                "event_date": "2025-02-15",
                "summary": None,
                "quote": "a total of 9 confirmed cases including 3 deaths have been reported",
            },
            {
                "event_type": "death_count",
                "confidence": 0.90,
                "location": "Mubende district, Uganda",
                "pathogen": "Sudan virus",
                "metric_name": "deaths",
                "metric_value": 3,
                "metric_unit": "deaths",
                "event_date": "2025-02-15",
                "summary": None,
                "quote": "9 confirmed cases including 3 deaths have been reported",
            },
        ],
    },
    input_tokens=350,
    output_tokens=180,
    model="gpt-4o-mini",
    raw_text='{"facts": [...]}',
)

# Response for chunk-who-sudan-001-t2 (table chunk - heading only, limited info)
SUDAN_TABLE_RESPONSE = LLMResponse(
    content={"facts": []},
    input_tokens=200,
    output_tokens=15,
    model="gpt-4o-mini",
    raw_text='{"facts": []}',
)

# Response for chunk-who-sudan-001-p3 (public health response - intervention)
SUDAN_RESPONSE_CHUNK = LLMResponse(
    content={
        "facts": [
            {
                "event_type": "intervention",
                "confidence": 0.80,
                "location": "Uganda",
                "pathogen": "Sudan virus",
                "metric_name": None,
                "metric_value": None,
                "metric_unit": None,
                "event_date": None,
                "summary": "Uganda activated national emergency operations centre and deployed rapid response teams.",
                "quote": "activated its national emergency operations centre and deployed rapid response teams",
            },
        ],
    },
    input_tokens=280,
    output_tokens=120,
    model="gpt-4o-mini",
    raw_text='{"facts": [...]}',
)

# Response for heading chunk (no facts)
EMPTY_RESPONSE = LLMResponse(
    content={"facts": []},
    input_tokens=150,
    output_tokens=15,
    model="gpt-4o-mini",
    raw_text='{"facts": []}',
)

# Response for risk assessment chunk (no facts)
RISK_ASSESSMENT_RESPONSE = LLMResponse(
    content={"facts": []},
    input_tokens=250,
    output_tokens=15,
    model="gpt-4o-mini",
    raw_text='{"facts": []}',
)


# ---- CDC H5N1 document responses ----

# Response for chunk-cdc-h5n1-001-p0 (prose with herd count)
H5N1_PROSE_RESPONSE = LLMResponse(
    content={
        "facts": [
            {
                "event_type": "case_count",
                "confidence": 0.88,
                "location": "United States",
                "pathogen": "H5N1",
                "metric_name": "affected_herds",
                "metric_value": 1043,
                "metric_unit": "herds",
                "event_date": "2026-01-30",
                "summary": None,
                "quote": "USDA has confirmed detections in 1,043 livestock herds spanning 16 states",
            },
        ],
    },
    input_tokens=300,
    output_tokens=140,
    model="gpt-4o-mini",
    raw_text='{"facts": [...]}',
)

# Response for the table chunk
H5N1_TABLE_RESPONSE = LLMResponse(
    content={"facts": []},
    input_tokens=200,
    output_tokens=15,
    model="gpt-4o-mini",
    raw_text='{"facts": []}',
)

# Response for human cases chunk
H5N1_HUMAN_CASES_RESPONSE = LLMResponse(
    content={
        "facts": [
            {
                "event_type": "case_count",
                "confidence": 0.85,
                "location": "United States",
                "pathogen": "H5N1",
                "metric_name": "human_cases",
                "metric_value": 47,
                "metric_unit": "cases",
                "event_date": "2026-01-30",
                "summary": None,
                "quote": "47 cases of H5N1 infection have been reported in humans in the United States",
            },
        ],
    },
    input_tokens=280,
    output_tokens=130,
    model="gpt-4o-mini",
    raw_text='{"facts": [...]}',
)


# ---- Reuters mpox document responses ----

MPOX_PHEIC_RESPONSE = LLMResponse(
    content={
        "facts": [
            {
                "event_type": "outbreak_declared",
                "confidence": 0.95,
                "location": "Central and East Africa",
                "pathogen": "mpox",
                "metric_name": None,
                "metric_value": None,
                "metric_unit": None,
                "event_date": None,
                "summary": "WHO declared mpox outbreak a PHEIC.",
                "quote": "declared the mpox outbreak in Central and East Africa a public health emergency of international concern",
            },
        ],
    },
    input_tokens=250,
    output_tokens=110,
    model="gpt-4o-mini",
    raw_text='{"facts": [...]}',
)

MPOX_DETAILS_RESPONSE = LLMResponse(
    content={"facts": []},
    input_tokens=260,
    output_tokens=15,
    model="gpt-4o-mini",
    raw_text='{"facts": []}',
)


# ---- Hallucination test response ----

# This response contains a quote that does NOT appear in any chunk text.
# Used to test that the hallucination guard drops the fact.
HALLUCINATED_QUOTE_RESPONSE = LLMResponse(
    content={
        "facts": [
            {
                "event_type": "case_count",
                "confidence": 0.80,
                "location": "Uganda",
                "pathogen": "Sudan virus",
                "metric_name": "confirmed_cases",
                "metric_value": 50,
                "metric_unit": "cases",
                "event_date": "2025-02-20",
                "summary": None,
                "quote": "As of 20 February 2025, 50 confirmed cases have been identified",
            },
        ],
    },
    input_tokens=300,
    output_tokens=100,
    model="gpt-4o-mini",
    raw_text='{"facts": [...]}',
)


# ---- Duplicate fact response (same fact as SUDAN_PROSE_RESPONSE) ----
# Used to test cross-document deduplication

DUPLICATE_SUDAN_CASE_COUNT = LLMResponse(
    content={
        "facts": [
            {
                "event_type": "case_count",
                "confidence": 0.88,
                "location": "Mubende district, Uganda",
                "pathogen": "Sudan virus",
                "metric_name": "confirmed_cases",
                "metric_value": 9,
                "metric_unit": "cases",
                "event_date": "2025-02-15",
                "summary": None,
                "quote": "a total of 9 confirmed cases including 3 deaths have been reported",
            },
        ],
    },
    input_tokens=320,
    output_tokens=140,
    model="gpt-4o-mini",
    raw_text='{"facts": [...]}',
)
