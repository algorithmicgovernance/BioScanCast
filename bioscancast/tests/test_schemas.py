import json
from dataclasses import asdict
from datetime import datetime

from bioscancast.schemas import Document, DocumentChunk, ChunkReference, InsightRecord


# ---------------------------------------------------------------------------
# Worked example 1: WHO Disease Outbreak News PDF — Sudan virus in Uganda
# ---------------------------------------------------------------------------

def test_who_sudan_virus_document_and_insight():
    heading_chunk = DocumentChunk(
        chunk_id="chunk-who-sudan-001-h0",
        chunk_index=0,
        text="Sudan virus disease — Uganda",
        chunk_type="heading",
        heading="Disease Outbreak News",
        page_number=1,
        token_count=8,
    )

    prose_chunk = DocumentChunk(
        chunk_id="chunk-who-sudan-001-p1",
        chunk_index=1,
        text=(
            "On 12 February 2025, the Ministry of Health of Uganda notified "
            "WHO of an outbreak of Sudan virus disease (SVD) in Mubende "
            "district. As of 15 February 2025, a total of 9 confirmed cases "
            "including 3 deaths have been reported. Active case search and "
            "contact tracing are ongoing in the affected district."
        ),
        chunk_type="prose",
        heading="Epidemiological summary > Country reports > Uganda",
        page_number=2,
        token_count=72,
    )

    table_chunk = DocumentChunk(
        chunk_id="chunk-who-sudan-001-t2",
        chunk_index=2,
        text="Cases by district (as of 15 February 2025)",
        chunk_type="table",
        heading="Epidemiological summary > Country reports > Uganda",
        page_number=4,
        table_data=[
            ["District", "Confirmed", "Probable", "Deaths"],
            ["Mubende", "9", "0", "3"],
            ["Kassanda", "0", "1", "0"],
        ],
        token_count=30,
    )

    doc = Document(
        id="doc-who-sudan-001",
        result_id="r-who-sudan-001",
        source_url="https://who.int/emergencies/disease-outbreak-news/item/2025-DON548",
        domain="who.int",
        fetched_at=datetime(2025, 2, 16, 8, 30, 0),
        document_type="pdf",
        status="success",
        canonical_url="https://who.int/emergencies/disease-outbreak-news/item/2025-DON548",
        title="Sudan virus disease — Uganda",
        published_date=datetime(2025, 2, 15),
        language="en",
        page_count=12,
        char_count=18_420,
        token_count=4_310,
        http_status=200,
        content_type="application/pdf",
        chunks=[heading_chunk, prose_chunk, table_chunk],
        extracted_tables=[
            [
                ["District", "Confirmed", "Probable", "Deaths"],
                ["Mubende", "9", "0", "3"],
                ["Kassanda", "0", "1", "0"],
            ],
        ],
        extracted_dates=["12 February 2025", "15 February 2025"],
    )

    insight = InsightRecord(
        id="ins-sudan-cases-001",
        question_id="q-sudan-outbreak-2025",
        event_type="case_count",
        confidence=0.92,
        location="Mubende district, Uganda",
        iso_country_code="UG",
        pathogen="Sudan virus",
        metric_name="confirmed_cases",
        metric_value=9.0,
        metric_unit="cases",
        event_date=datetime(2025, 2, 15),
        model="gpt-4o-2025-01-01",
        extracted_at=datetime(2025, 2, 16, 9, 0, 0),
        sources=[
            ChunkReference(
                document_id="doc-who-sudan-001",
                chunk_id="chunk-who-sudan-001-t2",
                source_url="https://who.int/emergencies/disease-outbreak-news/item/2025-DON548",
                quote="Mubende | 9 | 0 | 3",
            ),
        ],
    )

    # Document structure
    assert len(doc.chunks) == 3
    assert doc.chunks[0].chunk_type == "heading"
    assert doc.chunks[1].chunk_type == "prose"
    assert doc.chunks[2].chunk_type == "table"
    assert doc.chunks[2].table_data is not None
    assert len(doc.extracted_tables) == 1
    assert doc.page_count == 12
    assert doc.status == "success"

    # Insight structure
    assert insight.metric_value == 9.0
    assert insight.event_type == "case_count"
    assert insight.iso_country_code == "UG"
    assert insight.confidence >= 0.0 and insight.confidence <= 1.0

    # Provenance links
    assert len(insight.sources) >= 1
    assert insight.sources[0].document_id == doc.id
    assert insight.sources[0].chunk_id == doc.chunks[2].chunk_id


# ---------------------------------------------------------------------------
# Worked example 2: CDC HTML dashboard — H5N1 livestock herds in the US
# ---------------------------------------------------------------------------

def test_cdc_h5n1_document_and_insight():
    prose_chunk = DocumentChunk(
        chunk_id="chunk-cdc-h5n1-001-p0",
        chunk_index=0,
        text=(
            "Since early 2024, highly pathogenic avian influenza A(H5N1) has "
            "been detected in dairy cattle and poultry flocks across multiple "
            "US states. As of 30 January 2026, USDA has confirmed detections "
            "in 1,043 livestock herds spanning 16 states."
        ),
        chunk_type="prose",
        heading="Situation summary",
        token_count=58,
    )

    table_chunk = DocumentChunk(
        chunk_id="chunk-cdc-h5n1-001-t1",
        chunk_index=1,
        text="Confirmed H5N1 livestock herd detections by state",
        chunk_type="table",
        heading="Situation summary > State-level data",
        table_data=[
            ["State", "Herds affected", "Last detection"],
            ["Texas", "312", "2026-01-28"],
            ["California", "198", "2026-01-30"],
            ["Iowa", "87", "2026-01-25"],
        ],
        token_count=40,
    )

    doc = Document(
        id="doc-cdc-h5n1-001",
        result_id="r-cdc-h5n1-001",
        source_url="https://cdc.gov/bird-flu/situation-summary/index.html",
        domain="cdc.gov",
        fetched_at=datetime(2026, 1, 31, 14, 0, 0),
        document_type="html",
        status="success",
        canonical_url="https://cdc.gov/bird-flu/situation-summary/index.html",
        title="H5N1 Bird Flu: Current Situation Summary",
        published_date=datetime(2026, 1, 30),
        language="en",
        char_count=9_870,
        token_count=2_210,
        http_status=200,
        content_type="text/html; charset=utf-8",
        chunks=[prose_chunk, table_chunk],
        extracted_dates=["30 January 2026", "2026-01-28", "2026-01-30", "2026-01-25"],
    )

    insight = InsightRecord(
        id="ins-h5n1-herds-001",
        question_id="q-h5n1-us-herds-2026",
        event_type="case_count",
        confidence=0.85,
        location="United States",
        iso_country_code="US",
        pathogen="H5N1",
        metric_name="affected_herds",
        metric_value=1043.0,
        metric_unit="herds",
        event_date=datetime(2026, 1, 30),
        summary="1,043 confirmed H5N1-affected livestock herds across 16 US states as of 2026-01-30.",
        model="gpt-4o-2025-01-01",
        extracted_at=datetime(2026, 1, 31, 14, 30, 0),
        sources=[
            ChunkReference(
                document_id="doc-cdc-h5n1-001",
                chunk_id="chunk-cdc-h5n1-001-p0",
                source_url="https://cdc.gov/bird-flu/situation-summary/index.html",
                quote="USDA has confirmed detections in 1,043 livestock herds spanning 16 states",
            ),
            ChunkReference(
                document_id="doc-cdc-h5n1-001",
                chunk_id="chunk-cdc-h5n1-001-t1",
                source_url="https://cdc.gov/bird-flu/situation-summary/index.html",
                quote="State | Herds affected | Last detection",
            ),
        ],
    )

    # Document structure
    assert len(doc.chunks) == 2
    assert doc.chunks[0].chunk_type == "prose"
    assert doc.chunks[1].chunk_type == "table"
    assert doc.chunks[1].table_data is not None
    assert doc.page_count is None  # HTML, no page concept
    assert doc.status == "success"

    # Insight structure
    assert insight.metric_value == 1043.0
    assert insight.event_type == "case_count"
    assert insight.iso_country_code == "US"

    # Multiple provenance sources
    assert len(insight.sources) == 2
    assert insight.sources[0].document_id == doc.id
    assert insight.sources[1].document_id == doc.id


# ---------------------------------------------------------------------------
# Worked example 3: Reuters news article — WHO declares mpox PHEIC
# ---------------------------------------------------------------------------

def test_reuters_mpox_pheic_insight():
    prose_chunk_0 = DocumentChunk(
        chunk_id="chunk-reuters-mpox-001-p0",
        chunk_index=0,
        text=(
            "The World Health Organization on Wednesday declared the mpox "
            "outbreak in Central and East Africa a public health emergency "
            "of international concern, its highest level of alarm."
        ),
        chunk_type="prose",
        token_count=35,
    )

    prose_chunk_1 = DocumentChunk(
        chunk_id="chunk-reuters-mpox-001-p1",
        chunk_index=1,
        text=(
            "WHO Director-General Tedros Adhanom Ghebreyesus said the "
            "declaration reflected the rapid spread of clade Ib across "
            "several countries that had not previously reported mpox cases. "
            "The emergency committee convened on Tuesday and recommended the "
            "PHEIC designation by consensus."
        ),
        chunk_type="prose",
        token_count=52,
    )

    doc = Document(
        id="doc-reuters-mpox-001",
        result_id="r-reuters-mpox-001",
        source_url="https://reuters.com/world/africa/who-declares-mpox-public-health-emergency-2025-08-14",
        domain="reuters.com",
        fetched_at=datetime(2025, 8, 14, 19, 45, 0),
        document_type="html",
        status="success",
        title="WHO declares mpox outbreak a public health emergency",
        published_date=datetime(2025, 8, 14),
        language="en",
        char_count=3_420,
        token_count=810,
        http_status=200,
        content_type="text/html; charset=utf-8",
        chunks=[prose_chunk_0, prose_chunk_1],
        extracted_dates=["Wednesday", "Tuesday"],
    )

    insight = InsightRecord(
        id="ins-mpox-pheic-001",
        question_id="q-mpox-pheic-2025",
        event_type="outbreak_declared",
        confidence=0.88,
        location="Central and East Africa",
        pathogen="mpox",
        event_date=datetime(2025, 8, 14),
        summary=(
            "WHO declared the mpox outbreak in Central and East Africa a "
            "public health emergency of international concern (PHEIC) on "
            "14 August 2025, citing rapid spread of clade Ib."
        ),
        model="gpt-4o-2025-01-01",
        extracted_at=datetime(2025, 8, 14, 20, 0, 0),
        sources=[
            ChunkReference(
                document_id="doc-reuters-mpox-001",
                chunk_id="chunk-reuters-mpox-001-p0",
                source_url="https://reuters.com/world/africa/who-declares-mpox-public-health-emergency-2025-08-14",
                quote="declared the mpox outbreak ... a public health emergency of international concern",
            ),
        ],
    )

    # Document: prose-only, no tables
    assert len(doc.chunks) == 2
    assert all(c.chunk_type == "prose" for c in doc.chunks)
    assert doc.chunks[0].table_data is None

    # Insight: non-numeric, uses summary instead
    assert insight.metric_value is None
    assert insight.metric_name is None
    assert insight.summary is not None
    assert insight.event_type == "outbreak_declared"

    # Provenance present
    assert len(insight.sources) >= 1
    assert insight.sources[0].document_id == doc.id


# ---------------------------------------------------------------------------
# Round-trip serialisation via dataclasses.asdict + json
# ---------------------------------------------------------------------------

def test_document_round_trip_json():
    chunk = DocumentChunk(
        chunk_id="c0",
        chunk_index=0,
        text="Sample text.",
        chunk_type="prose",
        token_count=3,
    )
    doc = Document(
        id="doc-rt-001",
        result_id="r-rt-001",
        source_url="https://example.com/article",
        domain="example.com",
        fetched_at=datetime(2025, 6, 1, 12, 0, 0),
        document_type="html",
        status="success",
        chunks=[chunk],
    )

    d = asdict(doc)
    serialised = json.dumps(d, default=str)
    assert isinstance(serialised, str)

    loaded = json.loads(serialised)
    assert loaded["id"] == doc.id
    assert loaded["result_id"] == doc.result_id
    assert len(loaded["chunks"]) == 1
    assert loaded["chunks"][0]["chunk_id"] == "c0"

    # InsightRecord round-trip
    insight = InsightRecord(
        id="ins-rt-001",
        question_id="q1",
        event_type="case_count",
        confidence=0.75,
        metric_value=42.0,
        sources=[
            ChunkReference(
                document_id="doc-rt-001",
                chunk_id="c0",
                source_url="https://example.com/article",
                quote="Sample text.",
            ),
        ],
    )
    d2 = asdict(insight)
    serialised2 = json.dumps(d2, default=str)
    assert isinstance(serialised2, str)

    loaded2 = json.loads(serialised2)
    assert loaded2["id"] == insight.id
    assert loaded2["metric_value"] == 42.0
    assert len(loaded2["sources"]) == 1


# ---------------------------------------------------------------------------
# Provenance chain integrity
# ---------------------------------------------------------------------------

def test_insight_provenance_chain():
    chunk = DocumentChunk(
        chunk_id="c-prov-0",
        chunk_index=0,
        text="Relevant passage for extraction.",
        chunk_type="prose",
    )
    doc = Document(
        id="doc-prov-001",
        result_id="r-prov-001",
        source_url="https://who.int/example",
        domain="who.int",
        fetched_at=datetime(2025, 3, 1),
        document_type="html",
        status="success",
        chunks=[chunk],
    )
    insight = InsightRecord(
        id="ins-prov-001",
        question_id="q-prov-001",
        event_type="case_count",
        confidence=0.80,
        sources=[
            ChunkReference(
                document_id=doc.id,
                chunk_id=chunk.chunk_id,
                source_url=doc.source_url,
                quote="Relevant passage for extraction.",
            ),
        ],
    )

    # Full provenance chain is navigable
    assert insight.sources[0].document_id == doc.id
    assert insight.sources[0].chunk_id == doc.chunks[0].chunk_id
    assert insight.sources[0].source_url == doc.source_url
    assert insight.question_id == "q-prov-001"


# ---------------------------------------------------------------------------
# Failed document (fetch error)
# ---------------------------------------------------------------------------

def test_failed_document():
    doc = Document(
        id="doc-fail-001",
        result_id="r-fail-001",
        source_url="https://example.org/report.pdf",
        domain="example.org",
        fetched_at=datetime(2025, 4, 10, 7, 0, 0),
        document_type="pdf",
        status="failed",
        error_message="Connection timeout after 30s",
        http_status=None,
        content_type=None,
    )

    assert doc.status == "failed"
    assert doc.error_message is not None
    assert len(doc.chunks) == 0
    assert doc.title is None
    assert doc.page_count is None

    # Should still be serialisable
    d = asdict(doc)
    serialised = json.dumps(d, default=str)
    assert isinstance(serialised, str)
