"""Hand-built Document instances for insight stage testing.

These fixtures are the most important file in this branch for keeping
the insight stage independent of extraction.  They mirror the three
worked examples from the schema spec (test_schemas.py):

1. WHO Sudan virus PDF — heading, prose, table chunks
2. CDC H5N1 dashboard HTML — prose + table
3. Reuters mpox news article — prose only
4. A failed document for skip-testing

Each document includes realistic chunk text with planted facts so
retrieval and extraction tests have known ground truth.
"""

from datetime import datetime

from bioscancast.schemas import Document, DocumentChunk
from bioscancast.filtering.models import ForecastQuestion


# ---- Forecast Questions ----

QUESTION_SUDAN = ForecastQuestion(
    id="q-sudan-outbreak-2025",
    text="Will the Sudan virus outbreak in Uganda exceed 50 confirmed cases by March 2025?",
    created_at=datetime(2025, 2, 16),
    target_date=datetime(2025, 3, 31),
    region="Uganda",
    pathogen="Sudan virus",
    event_type="case_count",
)

QUESTION_H5N1 = ForecastQuestion(
    id="q-h5n1-us-herds-2026",
    text="Will H5N1 be detected in more than 1,500 US livestock herds by June 2026?",
    created_at=datetime(2026, 1, 31),
    target_date=datetime(2026, 6, 30),
    region="United States",
    pathogen="H5N1",
    event_type="case_count",
)

QUESTION_MPOX = ForecastQuestion(
    id="q-mpox-pheic-2025",
    text="Will WHO declare the mpox outbreak a public health emergency of international concern in 2025?",
    created_at=datetime(2025, 8, 10),
    target_date=datetime(2025, 12, 31),
    region=None,
    pathogen="mpox",
    event_type="outbreak_declared",
)


# ---- Document 1: WHO Sudan Virus PDF ----

WHO_SUDAN_CHUNKS = [
    DocumentChunk(
        chunk_id="chunk-who-sudan-001-h0",
        chunk_index=0,
        text="Sudan virus disease — Uganda",
        chunk_type="heading",
        heading="Disease Outbreak News",
        page_number=1,
        token_count=8,
    ),
    DocumentChunk(
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
    ),
    DocumentChunk(
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
    ),
    DocumentChunk(
        chunk_id="chunk-who-sudan-001-p3",
        chunk_index=3,
        text=(
            "The Government of Uganda has activated its national emergency "
            "operations centre and deployed rapid response teams to Mubende "
            "and neighbouring districts. WHO is supporting surveillance "
            "strengthening and laboratory diagnostics."
        ),
        chunk_type="prose",
        heading="Public health response",
        page_number=5,
        token_count=48,
    ),
    DocumentChunk(
        chunk_id="chunk-who-sudan-001-p4",
        chunk_index=4,
        text=(
            "Risk assessment: The risk is considered high at the national "
            "level due to the severity of Sudan virus disease and limited "
            "health system capacity in the affected district. The risk at "
            "the regional and global level is assessed as low."
        ),
        chunk_type="prose",
        heading="Risk assessment",
        page_number=7,
        token_count=55,
    ),
]

DOC_WHO_SUDAN = Document(
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
    chunks=WHO_SUDAN_CHUNKS,
    extracted_tables=[
        [
            ["District", "Confirmed", "Probable", "Deaths"],
            ["Mubende", "9", "0", "3"],
            ["Kassanda", "0", "1", "0"],
        ],
    ],
    extracted_dates=["12 February 2025", "15 February 2025"],
)


# ---- Document 2: CDC H5N1 Dashboard HTML ----

CDC_H5N1_CHUNKS = [
    DocumentChunk(
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
    ),
    DocumentChunk(
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
    ),
    DocumentChunk(
        chunk_id="chunk-cdc-h5n1-001-p2",
        chunk_index=2,
        text=(
            "Human cases remain rare. As of 30 January 2026, 47 cases of "
            "H5N1 infection have been reported in humans in the United States, "
            "mostly among poultry and dairy workers with direct animal contact."
        ),
        chunk_type="prose",
        heading="Human cases",
        token_count=45,
    ),
]

DOC_CDC_H5N1 = Document(
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
    chunks=CDC_H5N1_CHUNKS,
    extracted_dates=["30 January 2026", "2026-01-28", "2026-01-30", "2026-01-25"],
)


# ---- Document 3: Reuters mpox news article ----

REUTERS_MPOX_CHUNKS = [
    DocumentChunk(
        chunk_id="chunk-reuters-mpox-001-p0",
        chunk_index=0,
        text=(
            "The World Health Organization on Wednesday declared the mpox "
            "outbreak in Central and East Africa a public health emergency "
            "of international concern, its highest level of alarm."
        ),
        chunk_type="prose",
        token_count=35,
    ),
    DocumentChunk(
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
    ),
]

DOC_REUTERS_MPOX = Document(
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
    chunks=REUTERS_MPOX_CHUNKS,
    extracted_dates=["Wednesday", "Tuesday"],
)


# ---- Document 4: Failed document (for skip testing) ----

DOC_FAILED = Document(
    id="doc-fail-001",
    result_id="r-fail-001",
    source_url="https://example.org/report.pdf",
    domain="example.org",
    fetched_at=datetime(2025, 4, 10, 7, 0, 0),
    document_type="pdf",
    status="failed",
    error_message="Connection timeout after 30s",
)


# ---- All documents for convenience ----

ALL_DOCUMENTS = [DOC_WHO_SUDAN, DOC_CDC_H5N1, DOC_REUTERS_MPOX, DOC_FAILED]
SUCCESS_DOCUMENTS = [DOC_WHO_SUDAN, DOC_CDC_H5N1, DOC_REUTERS_MPOX]
