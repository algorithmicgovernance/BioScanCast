"""Prompt templates for the chunk extraction pass.

Key design principles:
- Extract only facts directly supported by the chunk text.
- Require a verbatim quote (<=200 chars) for each fact.
- Allow zero facts (common and valid for irrelevant chunks).
- Include the forecast question for context but forbid answering it.
- Brief cognitive bias warnings from the grant proposal's Table 5.

Note: full cognitive bias mitigations belong in the forecasting stage,
not here.  Insight extraction is neutral fact-finding.
"""

from __future__ import annotations

from bioscancast.schemas import Document, DocumentChunk
from bioscancast.stages.filtering.models import ForecastQuestion


EXTRACTION_SYSTEM_PROMPT = """\
You are a biosecurity fact extractor.  Your job is to extract \
structured factual claims from a document chunk that are relevant \
to a specific forecast question.

RULES:
1. Extract ONLY facts that are directly stated in or clearly supported \
by the chunk text.  Do NOT infer, speculate, or use outside knowledge.
2. For each fact, provide a verbatim quote from the chunk (max 200 \
characters) that supports the claim.  The quote must be an exact \
substring of the chunk text.  The quote MUST be the sentence (or \
sentence fragment) that carries the figure itself — it must contain \
the metric_value either as digits (e.g. "82"), as a number-word \
(e.g. "eighty-two", "a dozen"), or as a clear relative reference \
(e.g. "a quarter of the population"). A contextual or supporting \
sentence that mentions the topic but not the figure is NOT acceptable.
3. If the chunk contains no relevant facts, return an empty facts list. \
This is expected and common — most chunks are irrelevant.
4. Do NOT answer the forecast question.  Your job is fact extraction, \
not forecasting.
5. For event_date, use the most specific ISO date you can extract from \
the chunk and nothing more: ``YYYY-MM-DD`` when a day is given, \
``YYYY-MM`` when only a month is given (e.g. "January 2026"), or \
``YYYY`` when only a year is given. Do NOT invent a day-of-month when \
the chunk only mentions a month.
6. For metric_name, use one of these canonical snake_case values when \
applicable (this lets downstream dedup merge facts about the same \
metric across sources):
   - confirmed_cases       (the "confirmed" tier — lab-confirmed)
   - suspected_cases       (the "not-yet-confirmed" tier — covers \
"suspected", "probable", and "possible" reporting categories)
   - confirmed_or_probable_cases   (WHO/CDC's combined reporting bucket)
   - deaths                (lab-confirmed deaths)
   - suspected_deaths      (the "not-yet-confirmed" tier for deaths — \
covers "suspected", "probable", "under investigation" reporting)
   - hospitalizations
   - recoveries
   - vaccinations_administered
   - vaccine_doses_distributed
   - affected_herds         (animal disease — herds/farms affected)
   - affected_animals
   - new_outbreaks_declared
   - reproductive_number    (R0, Rt)
   - case_fatality_ratio
   If none of these fit, invent a short snake_case label. Do NOT put \
qualifiers (sex, age, sub-region, time-period like "weekly") in \
metric_name — capture those in `summary` or `location` instead. \
"cases", "reported cases", "total cases" all map to confirmed_cases. \
"suspected cases", "probable cases", "possible cases" all map to \
suspected_cases. "deaths" alone maps to deaths; "suspected deaths", \
"probable deaths", "deaths under investigation" map to suspected_deaths.
7. Be aware of cognitive biases that affect information processing:
   - Anchoring: do not over-weight the first number you encounter.
   - Availability: rare dramatic events are not necessarily more likely.
   - Overconfidence: if the chunk is ambiguous, lower your confidence.

OUTPUT: Return a JSON object with a "facts" array.  Each fact has the \
fields defined in the schema.  Return {"facts": []} if no relevant \
facts are found."""


def build_extraction_prompt(
    chunk: DocumentChunk,
    document: Document,
    question: ForecastQuestion,
) -> tuple[str, str, dict]:
    """Build the (system, user, json_schema) tuple for chunk extraction.

    Args:
        chunk: The specific chunk to extract from.
        document: Parent document (for metadata context).
        question: The forecast question driving extraction.

    Returns:
        Tuple of (system_prompt, user_prompt, json_schema).
    """
    # Build user prompt with context
    parts = [
        f"FORECAST QUESTION: {question.text}",
    ]
    if question.pathogen:
        parts.append(f"PATHOGEN: {question.pathogen}")
    if question.region:
        parts.append(f"REGION: {question.region}")

    parts.append("")
    parts.append(f"DOCUMENT: {document.title or document.source_url}")
    parts.append(f"SOURCE: {document.source_url}")
    if document.published_date:
        parts.append(f"PUBLISHED: {document.published_date.strftime('%Y-%m-%d')}")

    parts.append("")
    if chunk.heading:
        parts.append(f"SECTION: {chunk.heading}")
    parts.append(f"CHUNK TEXT:\n{chunk.text}")

    user_prompt = "\n".join(parts)

    return EXTRACTION_SYSTEM_PROMPT, user_prompt, EXTRACTION_JSON_SCHEMA


EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "enum": [
                            "case_count",
                            "death_count",
                            "outbreak_declared",
                            "intervention",
                            "policy_change",
                            "other",
                        ],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "location": {"type": ["string", "null"]},
                    "pathogen": {"type": ["string", "null"]},
                    "metric_name": {"type": ["string", "null"]},
                    "metric_value": {"type": ["number", "null"]},
                    "metric_unit": {"type": ["string", "null"]},
                    "event_date": {"type": ["string", "null"]},
                    "summary": {"type": ["string", "null"]},
                    "quote": {"type": "string"},
                },
                "required": [
                    "event_type",
                    "confidence",
                    "location",
                    "pathogen",
                    "metric_name",
                    "metric_value",
                    "metric_unit",
                    "event_date",
                    "summary",
                    "quote",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}
