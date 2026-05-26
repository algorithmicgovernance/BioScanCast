"""Tests for insight stage chunk extraction.

All tests use FakeLLMClient — no network calls, no real OpenAI imports.
"""

import pytest

from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.insight.extraction.chunk_extractor import (
    extract_facts_from_chunk,
    _resolve_country_code,
    _normalize_whitespace,
    _quote_matches,
)

from bioscancast.tests.fixtures.insight.synthetic_documents import (
    DOC_WHO_SUDAN,
    DOC_CDC_H5N1,
    QUESTION_SUDAN,
    QUESTION_H5N1,
)
from bioscancast.tests.fixtures.insight.fake_llm_responses import (
    SUDAN_PROSE_RESPONSE,
    HALLUCINATED_QUOTE_RESPONSE,
    H5N1_PROSE_RESPONSE,
    EMPTY_RESPONSE,
)


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_extract_facts_from_sudan_prose():
    """Extracting from the Sudan virus prose chunk should produce valid
    InsightRecords with correct provenance."""
    client = FakeLLMClient([SUDAN_PROSE_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[1]  # prose chunk with case count

    records, response = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    assert len(records) == 2  # case_count + death_count

    case_record = records[0]
    assert case_record.event_type == "case_count"
    assert case_record.metric_value == 9.0
    assert case_record.question_id == QUESTION_SUDAN.id

    # Provenance
    assert len(case_record.sources) == 1
    assert case_record.sources[0].document_id == DOC_WHO_SUDAN.id
    assert case_record.sources[0].chunk_id == chunk.chunk_id
    assert case_record.sources[0].source_url == DOC_WHO_SUDAN.source_url

    # Quote should be a substring of the chunk text (hallucination guard passed)
    quote = case_record.sources[0].quote
    assert _normalize_whitespace(quote) in _normalize_whitespace(chunk.text)


def test_extract_facts_country_normalization():
    """Country names should be normalized to ISO codes."""
    client = FakeLLMClient([SUDAN_PROSE_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[1]

    records, _ = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    # "Mubende district, Uganda" -> "UG"
    assert records[0].iso_country_code == "UG"


def test_extract_facts_from_h5n1():
    """H5N1 extraction should produce correct metric and country code."""
    client = FakeLLMClient([H5N1_PROSE_RESPONSE])
    chunk = DOC_CDC_H5N1.chunks[0]

    records, _ = extract_facts_from_chunk(
        chunk, DOC_CDC_H5N1, QUESTION_H5N1, client, model="gpt-4o-mini"
    )

    assert len(records) == 1
    assert records[0].metric_value == 1043.0
    assert records[0].metric_name == "affected_herds"
    assert records[0].iso_country_code == "US"


def test_extract_empty_response():
    """A chunk with no relevant facts should return an empty list."""
    client = FakeLLMClient([EMPTY_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[0]  # heading chunk

    records, response = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    assert records == []
    assert response.input_tokens > 0  # Budget still tracked


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------


def test_hallucination_guard_drops_fake_quote():
    """The hallucination guard must drop facts whose quote does not
    appear in the chunk text.  This test should FAIL if the guard
    is removed."""
    client = FakeLLMClient([HALLUCINATED_QUOTE_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[1]  # prose chunk

    records, response = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    # The hallucinated quote "As of 20 February 2025, 50 confirmed cases
    # have been identified" does NOT appear in the chunk text.
    # The guard should drop it.
    assert len(records) == 0

    # But the LLM response was still consumed (budget tracking)
    assert response.input_tokens == 300


def test_hallucination_guard_passes_valid_quote():
    """Valid quotes should pass the hallucination guard."""
    client = FakeLLMClient([SUDAN_PROSE_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[1]

    records, _ = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    # Both facts have valid quotes
    assert len(records) == 2


# ---------------------------------------------------------------------------
# Country code resolution
# ---------------------------------------------------------------------------


def test_resolve_country_code_simple():
    assert _resolve_country_code("Uganda") == "UG"
    assert _resolve_country_code("United States") == "US"
    assert _resolve_country_code("usa") == "US"


def test_resolve_country_code_with_region():
    assert _resolve_country_code("Mubende district, Uganda") == "UG"
    assert _resolve_country_code("Texas") == "US"


def test_resolve_country_code_unknown():
    assert _resolve_country_code("Unknown Planet") is None
    assert _resolve_country_code(None) is None
    assert _resolve_country_code("") is None


@pytest.mark.parametrize(
    "location,expected",
    [
        # All 249 ISO entries are covered by pycountry — spot-check the
        # ones that show up in the 6 real biosecurity test documents.
        ("Comoros", "KM"),
        ("Bulgaria", "BG"),
        ("Austria", "AT"),
        ("Ireland", "IE"),
        ("Portugal", "PT"),
        ("Sweden", "SE"),
        ("Romania", "RO"),
        ("Netherlands", "NL"),
        ("Madagascar", "MG"),
        ("Mozambique", "MZ"),
        ("Côte d'Ivoire", "CI"),
        ("Côte d’Ivoire", "CI"),  # curly apostrophe
        # Alpha-2 / alpha-3 codes
        ("DE", "DE"),
        ("DEU", "DE"),
    ],
)
def test_resolve_country_code_pycountry_lookups(location, expected):
    assert _resolve_country_code(location) == expected


@pytest.mark.parametrize(
    "location,expected",
    [
        # Aliases that pycountry doesn't catch directly
        ("UK", "GB"),
        ("U.K.", "GB"),
        ("England", "GB"),
        ("Scotland", "GB"),
        ("DRC", "CD"),
        ("DR Congo", "CD"),
        ("Democratic Republic of the Congo", "CD"),
        ("Republic of the Congo", "CG"),
        # Bare "Congo" intentionally resolves to CG (Republic of) via
        # pycountry's canonical name — biosecurity reporting that means
        # DRC should use one of the DRC aliases.
        ("Congo", "CG"),
        ("UAE", "AE"),
        ("Russia", "RU"),
        ("Burma", "MM"),
        ("Myanmar", "MM"),
        ("Ivory Coast", "CI"),
        ("North Korea", "KP"),
        ("South Korea", "KR"),
    ],
)
def test_resolve_country_code_aliases(location, expected):
    assert _resolve_country_code(location) == expected


@pytest.mark.parametrize(
    "state",
    [
        "California", "Iowa", "New Mexico", "Utah", "Texas",
        "New York", "Florida", "Washington",
    ],
)
def test_resolve_country_code_us_states_resolve_to_us(state):
    assert _resolve_country_code(state) == "US"


@pytest.mark.parametrize(
    "region",
    [
        "Africa", "Asia", "Europe", "Americas", "Oceania",
        "European Region", "African Region",
        "Region of the Americas", "Western Pacific Region",
        "South-East Asia Region", "Eastern Mediterranean Region",
        "EU/EEA", "EU", "EEA",
        "World", "global",
    ],
)
def test_resolve_country_code_multi_country_regions_return_none(region):
    assert _resolve_country_code(region) is None


@pytest.mark.parametrize(
    "location,expected",
    [
        # Compound location strings — the resolver should try the last
        # segment after a comma.
        ("Mubende district, Uganda", "UG"),
        ("Lea County, New Mexico", "US"),
        ("Ngazidja region, Comoros", "KM"),
        ("Lyon, France", "FR"),
        # When the last segment is a region label, the resolver should
        # return None rather than guessing.
        ("Some city, European Region", None),
    ],
)
def test_resolve_country_code_compound_locations(location, expected):
    assert _resolve_country_code(location) == expected


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


def test_response_returned_for_budget_tracking():
    """extract_facts_from_chunk must always return the LLMResponse
    so the pipeline can record it in the budget tracker."""
    client = FakeLLMClient([EMPTY_RESPONSE])
    chunk = DOC_WHO_SUDAN.chunks[0]

    _, response = extract_facts_from_chunk(
        chunk, DOC_WHO_SUDAN, QUESTION_SUDAN, client, model="gpt-4o-mini"
    )

    assert response is not None
    assert response.input_tokens == 150
    assert response.output_tokens == 15
    assert response.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Hallucination guard — layered match behaviour
# ---------------------------------------------------------------------------

# Each tuple: (label, chunk_text, quote, should_match)
#
# These cases come from live-LLM observations on real WHO/CDC/ECDC documents.
# The strict substring guard rejected ~85% of real factual quotes due to
# minor punctuation/unicode drift; the looser layered guard must keep
# accepting wholesale fabrications while accepting the real-but-drifted
# variants below.

_LAYER1_NFKC_CASES = [
    (
        "curly-apos source vs straight-apos quote",
        "Côte d’Ivoire reported four confirmed cases each in January",
        "Côte d'Ivoire reported four confirmed cases each in January",
        True,
    ),
    (
        "non-breaking space in source number",
        "30 EU/EEA Member States reported a total of 4 623 cases of measles.",
        "30 EU/EEA Member States reported a total of 4 623 cases of measles",
        True,
    ),
    (
        "em-dash in source vs hyphen in quote",
        "Measles—Multi-country—Monitoring European outbreaks",
        "Measles-Multi-country-Monitoring European outbreaks",
        # The typography fold step maps em-dash to hyphen.
        True,
    ),
    (
        "newline in source vs space in quote",
        "Democratic Republic of the Congo\n6 543",
        "Democratic Republic of the Congo 6 543",
        True,
    ),
]


_LAYER2_TERMINAL_PUNCTUATION_CASES = [
    (
        "comma in source becomes period in quote",
        "...reported by Italy (63), Spain (36), France (16) and Poland (five).",
        "The highest case counts were reported by Italy (63).",
        # Quote prefix not in source — should reject
        False,
    ),
    (
        "comma in source becomes period in quote — full prefix",
        "The highest case counts were reported by Italy (63), Spain (36), France (16) and Poland (five).",
        "The highest case counts were reported by Italy (63).",
        True,
    ),
    (
        "no terminator in source vs period in quote",
        "Spain reported 97 cases of measles from 1 January to 12 April 2026 according to ECDC",
        "Spain reported 97 cases of measles from 1 January to 12 April 2026.",
        True,
    ),
]


_LAYER3_WRAPPING_PUNCTUATION_CASES = [
    (
        "parens around acronym in source dropped in quote",
        "f Health (NMDOH) eventually reported 99 outbreak-related measles cases, approximately one half.",
        "NMDOH eventually reported 99 outbreak-related measles cases.",
        True,
    ),
    (
        "double quotes in source dropped in quote",
        "The CDC said “we are responding” to the outbreak.",
        "The CDC said we are responding to the outbreak.",
        True,
    ),
    (
        "square brackets in source dropped in quote",
        "the result [12] shows a clear trend in cases.",
        "the result 12 shows a clear trend in cases.",
        True,
    ),
]


_HALLUCINATION_CASES = [
    (
        "fabricated word inserted into list",
        "Ghana and Liberia have reported human mpox due to clade IIa MPXV.",
        "Ghana, Atlantis, and Liberia have reported human mpox due to clade IIa MPXV.",
        False,
    ),
    (
        "wholesale fabrication",
        "Some real chunk content about measles cases in Utah.",
        "THIS QUOTE WAS INVENTED BY THE MODEL AND APPEARS NOWHERE.",
        False,
    ),
    (
        "synthesised prefix bolted onto a fragment",
        # Model takes a prefix from sentence A and bolts it onto a fragment of sentence B
        "Italy reported 63 cases. Spain reported 36 cases.",
        "Italy reported 36 cases.",
        False,
    ),
    (
        "wrong number",
        "Spain reported 36 cases yesterday.",
        "Spain reported 363 cases yesterday.",
        False,
    ),
    (
        "empty quote",
        "Real chunk content.",
        "",
        False,
    ),
    (
        "whitespace-only quote",
        "Real chunk content.",
        "   \n\t  ",
        False,
    ),
]


@pytest.mark.parametrize(
    "label,chunk_text,quote,should_match",
    _LAYER1_NFKC_CASES + _LAYER2_TERMINAL_PUNCTUATION_CASES + _LAYER3_WRAPPING_PUNCTUATION_CASES,
)
def test_quote_matches_accepts_real_quotes_with_normalisation_drift(
    label, chunk_text, quote, should_match
):
    """Real factual quotes with NFKC / terminal-punctuation / wrapping-
    punctuation drift should be accepted by the layered guard."""
    result = _quote_matches(quote, chunk_text)
    if should_match:
        assert result is not None, (
            f"{label}: expected match, got None. quote={quote!r}"
        )
        # The returned canonical form must itself be a substring of the
        # *normalised* chunk text — that's the invariant the guard
        # guarantees to downstream consumers.
        from bioscancast.insight.extraction.chunk_extractor import (
            _normalize_for_match,
            _WRAPPING_PUNCT_RE,
        )
        import re
        norm_chunk = _normalize_for_match(chunk_text)
        unwrap_chunk = re.sub(
            r"\s+", " ", _WRAPPING_PUNCT_RE.sub("", norm_chunk)
        ).strip()
        assert result in norm_chunk or result in unwrap_chunk, (
            f"{label}: canonical form {result!r} not in chunk after "
            "normalisation"
        )
    else:
        assert result is None, f"{label}: expected reject, got {result!r}"


@pytest.mark.parametrize(
    "label,chunk_text,quote,should_match", _HALLUCINATION_CASES
)
def test_quote_matches_rejects_hallucinations(
    label, chunk_text, quote, should_match
):
    """Content insertions, fabrications, wrong numbers, and synthesised
    quotes must all be rejected even by the looser layered guard."""
    result = _quote_matches(quote, chunk_text)
    assert result is None, (
        f"{label}: expected reject, got {result!r} (quote={quote!r})"
    )


def test_quote_matches_returns_canonical_form_not_raw_quote():
    """When a quote matches via terminal-punctuation strip, the canonical
    returned form should be the stripped version — not the model's raw
    output — so downstream consumers always see a verbatim chunk
    substring."""
    chunk = "Italy reported 63 cases, Spain reported 36 cases."
    quote = "Italy reported 63 cases."  # Model added period
    result = _quote_matches(quote, chunk)
    assert result == "Italy reported 63 cases"
    # Round-trip: the canonical form must be a substring of the
    # normalised chunk
    assert result in _normalize_whitespace(chunk)
