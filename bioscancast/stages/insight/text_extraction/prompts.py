"""Prompt templates for the chunk extraction pass.

Key design principles:
- Extract only facts directly supported by the chunk text.
- Require a verbatim quote (<=200 chars) for each fact.
- Allow zero facts (common and valid for irrelevant chunks).
- Include the forecast question for context but forbid answering it.
- Only OBSERVED counts become case_count/death_count; projected, modeled,
  and off-scope numbers are kept as context, not observed metrics, and
  confidence tracks how well a fact's scope matches the question. This
  guards the "lurch" failure where a projection or wrong-scope figure was
  filed as an observed count at high confidence (rules 7-9); the
  bioscancast/tests/test_insight_extraction_traps.py trap-set pins it.
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
7. OBSERVED vs NON-OBSERVED. event_type ``case_count`` and ``death_count`` \
are for OBSERVED, actually-reported counts ONLY. NEVER assign them to a \
number that is projected, modeled, forecasted, simulated, estimated, \
hypothetical, or a target/threshold — even when the number is stated \
plainly (e.g. "could see as many as 10,000 cases by year-end", "one in 20 \
simulations projected ..."). For such a number, set event_type ``other``, \
leave metric_value null, and describe it in ``summary`` (note it is a \
projection/model output and give its horizon and scenario if stated).
8. SCOPE MATCH. A number pertains to the question only if its metric, its \
geography, AND its time window all match the question's. Do NOT relabel a \
number to the question's scope: if the geography differs (e.g. a GLOBAL \
cumulative total against a United-States question), set ``location`` to the \
number's true scope (e.g. "Global"), never the question's region, and note \
the period (e.g. "cumulative since 2003") in ``summary``. A sub-national \
figure (one state/district) keeps its sub-national ``location`` and is NOT \
the national total. A weekly / "new this week" increment must say so in \
``summary``; it is NOT the cumulative total.
9. CONFIDENCE reflects SCOPE-MATCH, not textual presence. Set ``confidence`` \
by how well the fact's metric/geography/time-window match the question, NOT \
by whether the number appears in the text. A number present verbatim but of \
uncertain or mismatched scope gets LOW confidence (<= 0.5). Reserve high \
confidence (> 0.85) for numbers whose metric, geography, and window clearly \
match the question.
10. COUNT BASIS. For every case/death-type metric, set ``count_basis`` to \
exactly one of: ``cumulative`` (a running total to date — "cumulative total", \
"since the start", "to date", "so far"), ``incident`` (new counts during a \
period — "new", "reported this week", "in January"), ``active`` (currently \
active/ongoing — "active cases", "currently under isolation"), ``prevalence`` \
(point-prevalent cases), or ``unknown``. This is the KIND of number, \
independent of its value. Do NOT guess: if the chunk gives no explicit cue, \
use ``unknown``. Ratios and rates that are not counts (e.g. \
``case_fatality_ratio``, ``reproductive_number``) use ``unknown``.
11. TIME WINDOW. Set ``time_window`` to the reporting period ONLY for \
``incident`` counts and ONLY when the chunk states it: ``day``, ``week``, \
``month``, or ``year``; otherwise ``unknown``. Cumulative, active, and \
prevalence counts always use ``unknown`` (they have no window). This \
complements rule 8: a weekly increment is ``count_basis=incident``, \
``time_window=week`` — and still note "new this week" in ``summary``.
12. SURVEILLANCE METHOD. Set ``surveillance_method`` to a surveillance or \
ascertainment method ONLY when it is explicitly stated (e.g. "laboratory \
surveillance", "syndromic surveillance", "enhanced surveillance", "passive \
reporting"). Otherwise set it to null. NEVER infer it.
13. DATA QUALITY. Set ``data_quality`` to a short verbatim-ish note ONLY when \
the chunk explicitly states a caveat about how the numbers relate to reality: \
under-reporting ("likely underreported", "true number is higher"), limited \
testing/ascertainment ("testing capacity limited", "many mild cases not \
captured"), reporting lag/delay, a suspected-vs-confirmed definition issue, or \
a surveillance/case-definition change. Otherwise set it to null. Do NOT infer \
under-reporting from a number alone, and do NOT restate the metric here — this \
field is only for an explicit data-quality statement present in the text.
14. Be aware of cognitive biases that affect information processing:
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
                    # count_basis / time_window are non-nullable strings with an
                    # explicit "unknown" member rather than nullable enums: under
                    # OpenAI strict json-schema a nullable enum must also list
                    # null, and "unknown" already expresses "not stated" more
                    # legibly downstream. surveillance_method stays nullable
                    # free-text (no controlled vocabulary).
                    "count_basis": {
                        "type": "string",
                        "enum": [
                            "cumulative",
                            "incident",
                            "active",
                            "prevalence",
                            "unknown",
                        ],
                    },
                    "time_window": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year", "unknown"],
                    },
                    "surveillance_method": {"type": ["string", "null"]},
                    "data_quality": {"type": ["string", "null"]},
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
                    "count_basis",
                    "time_window",
                    "surveillance_method",
                    "data_quality",
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
