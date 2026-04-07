# bioscancast/schemas

Shared dataclass definitions for the data objects passed between the
**extraction**, **insight**, and (future) **forecasting** pipeline stages.

These schemas form the contract that lets extraction and insight be
developed on separate branches without coordination beyond this package.

## Dataclass reference

| Class | Module | Description |
|-------|--------|-------------|
| `DocumentChunk` | `schemas.document` | One content chunk (prose, table, heading, etc.) within a document. |
| `Document` | `schemas.document` | A fetched and chunked document — output of the extraction stage. |
| `ChunkReference` | `schemas.insight_record` | Provenance link from an insight back to a specific chunk and source URL. |
| `InsightRecord` | `schemas.insight_record` | An atomic factual claim extracted from document chunks — output of the insight stage. |

All four types are re-exported from `bioscancast.schemas`:

```python
from bioscancast.schemas import Document, DocumentChunk, ChunkReference, InsightRecord
```

## Stage boundary

```
filtering stage                 extraction stage                insight stage
───────────────                 ────────────────                ─────────────
FilteredDocument  ──────────►   Document            ──────────► InsightRecord
                                 ├─ DocumentChunk[]               ├─ ChunkReference[]
                                 └─ (tables, dates)               └─ (structured fact)
```

* **Extraction** receives `FilteredDocument` objects, fetches the URL,
  parses the content, and produces one `Document` per input.
  `Document.result_id` links back to `FilteredDocument.result_id`.

* **Insight** receives `Document` objects and a `ForecastQuestion`,
  extracts structured facts, and produces `InsightRecord` objects.
  Each `InsightRecord` cites its sources via `ChunkReference` entries
  that point to `Document.id` / `DocumentChunk.chunk_id`.

* **Forecasting** (future) will consume `InsightRecord` objects.
  Forecast output schemas are out of scope and will be added later.

## Provenance chain

Every forecast must be auditable back to a source URL and passage:

```
ForecastQuestion.id      <──  InsightRecord.question_id
Document.result_id       ──>  FilteredDocument.result_id
Document.id              <──  InsightRecord.sources[].document_id
DocumentChunk.chunk_id   <──  InsightRecord.sources[].chunk_id
source URL               <──  InsightRecord.sources[].source_url
verbatim quote           <──  InsightRecord.sources[].quote
```

## Serialisation

All schemas are JSON-serialisable via the standard library:

```python
import json
from dataclasses import asdict

json.dumps(asdict(doc), default=str)   # default=str handles datetime
```

## Extending the schemas

* **Add fields** with `Optional[...] = None` or `List[...] = field(default_factory=list)`
  so existing serialised data remains loadable.
* **Extend string enumerations** (e.g. `event_type`, `chunk_type`) freely —
  they are plain strings, not Python Enums.
* **Do not remove or rename** existing fields without coordinating across
  extraction and insight branches.
* If you need a new top-level schema (e.g. `ForecastOutput`), add a new
  module in this package and re-export from `__init__.py`.

## Known issue

The project README describes an aspirational `stages/` directory layout
that does not match the current flat layout (e.g. `bioscancast/filtering/`).
This schemas package follows the actual layout.  Reconciling the README
is tracked separately and out of scope for this change.
