from datetime import datetime, timezone

from bioscancast.stages.filtering.models import ForecastQuestion
from bioscancast.stages.searching.source_lookup import (
    lookup_pathogen_sources,
    lookup_yaml_sources,
    resolve_pathogen_entries,
    _resolve_pathogen_key,
    _load_sources_yaml,
)


# The YAML-backed registry makes the caller supply the *route* (family);
# pathogen -> pathogen-key within that route is resolved by
# _resolve_pathogen_key. Routes used below: h5n1/"bird flu" -> respiratory,
# mpox/monkeypox -> pox_re_emerging_viruses, ebola/marburg -> hemorrhagic.
def _make_question(**overrides):
    defaults = {
        "id": "Q001",
        "text": "Will H5N1 cause more than 100 human cases?",
        "created_at": datetime(2025, 1, 15, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return ForecastQuestion(**defaults)


class TestYamlSourceLookup:
    def test_known_pathogen_returns_results(self):
        q = _make_question(pathogen="h5n1")
        results = lookup_yaml_sources(q, "respiratory")
        assert len(results) > 0
        for r in results:
            assert r.retrieval_reason == "dashboard_lookup"
            assert r.rank == 0
            assert r.engine == "dashboard"
            assert r.question_id == "Q001"
            assert r.query_id == "dashboard_Q001"

    def test_mpox_returns_results(self):
        q = _make_question(pathogen="mpox")
        results = lookup_yaml_sources(q, "pox_re_emerging_viruses")
        assert len(results) > 0

    def test_unknown_pathogen_under_family_returns_empty(self):
        # An unrecognised pathogen under a family route returns nothing rather
        # than flattening the whole family: injecting ebola + marburg dashboards
        # for, say, a Lassa question would feed the insight stage wrong-pathogen
        # case counts. The pipeline falls back to general_sources instead.
        # (issue #41, finding #1)
        q = _make_question(pathogen="unknownvirus123")
        results = lookup_yaml_sources(q, "hemorrhagic")
        assert results == []

    def test_no_pathogen_returns_empty(self):
        # No pathogen + a pathogen-specific route yields nothing; only the
        # general_sources route is pathogen-independent.
        q = _make_question(pathogen=None)
        results = lookup_yaml_sources(q, "hemorrhagic")
        assert results == []

    def test_general_sources_route_ignores_pathogen(self):
        q = _make_question(pathogen=None)
        results = lookup_yaml_sources(q, "general_sources")
        assert len(results) > 0

    def test_case_insensitive(self):
        q = _make_question(pathogen="H5N1")
        results = lookup_yaml_sources(q, "respiratory")
        assert len(results) > 0

    def test_multiword_pathogen_routes_via_substring(self):
        # "Marburg Virus Disease" -> pathogen "marburg virus disease" must still
        # resolve to the "marburg" key within the hemorrhagic family.
        canonical = lookup_yaml_sources(
            _make_question(pathogen="marburg"), "hemorrhagic"
        )
        multiword = lookup_yaml_sources(
            _make_question(pathogen="marburg virus disease"), "hemorrhagic"
        )
        assert len(multiword) > 0
        assert [r.url for r in multiword] == [r.url for r in canonical]

    def test_alias_routes_to_canonical(self):
        # "monkeypox" -> "mpox"; "bird flu" -> "h5n1".
        monkeypox = lookup_yaml_sources(
            _make_question(pathogen="monkeypox"), "pox_re_emerging_viruses"
        )
        mpox = lookup_yaml_sources(
            _make_question(pathogen="mpox"), "pox_re_emerging_viruses"
        )
        assert len(monkeypox) > 0
        assert [r.url for r in monkeypox] == [r.url for r in mpox]
        assert (
            len(lookup_yaml_sources(_make_question(pathogen="bird flu"), "respiratory"))
            > 0
        )

    def test_results_have_required_fields(self):
        q = _make_question(pathogen="ebola")
        results = lookup_yaml_sources(q, "hemorrhagic")
        assert len(results) > 0
        for r in results:
            assert r.url is not None
            assert r.canonical_url is not None
            assert r.domain != ""
            assert r.source_tier in {"official", "academic", "trusted_media", "ngo", "unknown"}
            assert 0.0 <= r.domain_score <= 1.0
            assert r.freshness_score == 1.0
            assert r.search_stage_score == 0.0  # computed later by pipeline


class TestPathogenFirstLookup:
    """Deterministic pathogen-first resolution scans *all* families for
    question.pathogen — no LLM route required (issue #41)."""

    def test_resolves_across_families_without_route(self):
        results = lookup_pathogen_sources(_make_question(pathogen="mpox"))
        assert len(results) > 0
        for r in results:
            assert r.retrieval_reason == "dashboard_lookup"

    def test_unset_pathogen_returns_empty(self):
        assert lookup_pathogen_sources(_make_question(pathogen=None)) == []
        assert resolve_pathogen_entries(_make_question(pathogen=None)) == []

    def test_unrecognised_pathogen_returns_empty(self):
        # No family match -> empty, so the pipeline falls back to general.
        assert lookup_pathogen_sources(_make_question(pathogen="novel pathogen")) == []
        assert resolve_pathogen_entries(_make_question(pathogen="unknownvirus123")) == []

    def test_h5n1_mirrored_family_deduplicated(self):
        # h5n1 is intentionally mirrored under `respiratory` and
        # `animal_spillover`; a cross-family scan must not return duplicates.
        entries = resolve_pathogen_entries(_make_question(pathogen="h5n1"))
        ids = [e.get("id") for e in entries]
        assert ids == list(dict.fromkeys(ids)), f"duplicate entries: {ids}"
        urls = [e.get("url") for e in entries]
        assert len(urls) == len(set(urls))

    def test_alias_resolves_without_route(self):
        # Free-text topic strings from the benchmark loader must resolve.
        for text, canonical in [
            ("avian influenza h5", "h5n1"),
            ("sudan virus disease", "ebola"),
            ("sars-cov-2", "covid-19"),
            ("poliovirus", "polio"),
        ]:
            got = {r.url for r in lookup_pathogen_sources(_make_question(pathogen=text))}
            want = {r.url for r in lookup_pathogen_sources(_make_question(pathogen=canonical))}
            assert got == want and got, f"{text!r} did not resolve to {canonical!r}"

    def test_pathogen_first_excludes_general_sources(self):
        # Pathogen hit must not pull in the general baseline feeds.
        general_urls = {
            r.url for r in lookup_yaml_sources(_make_question(), "general_sources")
        }
        pathogen_urls = {
            r.url for r in lookup_pathogen_sources(_make_question(pathogen="mpox"))
        }
        assert pathogen_urls.isdisjoint(general_urls)


class TestResolvePathogenKey:
    def test_exact_alias_and_substring(self):
        cfg = _load_sources_yaml()
        respiratory = cfg["specific_pathogen_sources"]["respiratory"]
        hemorrhagic = cfg["specific_pathogen_sources"]["hemorrhagic"]
        assert _resolve_pathogen_key("h5n1", respiratory) == "h5n1"
        assert _resolve_pathogen_key("bird flu", respiratory) == "h5n1"  # alias
        assert _resolve_pathogen_key("marburg virus disease", hemorrhagic) == "marburg"  # substring
        assert _resolve_pathogen_key("unknownvirus123", hemorrhagic) is None
