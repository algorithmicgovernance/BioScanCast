from datetime import datetime, timezone

from bioscancast.stages.filtering.models import ForecastQuestion
from bioscancast.stages.searching.source_lookup import (
    lookup_yaml_sources,
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

    def test_unknown_pathogen_falls_back_to_family(self):
        # An unrecognised pathogen under a family route falls back to the whole
        # family (see _resolve_entries). Intentional best-effort behaviour;
        # flagged for review in the route-gating follow-up issue.
        q = _make_question(pathogen="unknownvirus123")
        results = lookup_yaml_sources(q, "hemorrhagic")
        assert len(results) > 0

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


class TestResolvePathogenKey:
    def test_exact_alias_and_substring(self):
        cfg = _load_sources_yaml()
        respiratory = cfg["specific_pathogen_sources"]["respiratory"]
        hemorrhagic = cfg["specific_pathogen_sources"]["hemorrhagic"]
        assert _resolve_pathogen_key("h5n1", respiratory) == "h5n1"
        assert _resolve_pathogen_key("bird flu", respiratory) == "h5n1"  # alias
        assert _resolve_pathogen_key("marburg virus disease", hemorrhagic) == "marburg"  # substring
        assert _resolve_pathogen_key("unknownvirus123", hemorrhagic) is None

    def test_bare_h5_and_h5nx_resolve_to_h5n1(self):
        # Unspecified-N inputs intentionally fall back to the h5n1 curated set.
        cfg = _load_sources_yaml()
        respiratory = cfg["specific_pathogen_sources"]["respiratory"]
        assert _resolve_pathogen_key("h5", respiratory) == "h5n1"
        assert _resolve_pathogen_key("h5nx", respiratory) == "h5n1"
        assert _resolve_pathogen_key("avian influenza", respiratory) == "h5n1"

    def test_non_h5n1_subtypes_do_not_resolve_to_h5n1(self):
        # Explicit non-H5N1 H5 subtypes must not be silently routed to the
        # H5N1 source set (issue #58).
        cfg = _load_sources_yaml()
        respiratory = cfg["specific_pathogen_sources"]["respiratory"]
        for subtype in ("h5n5", "h5n8", "h5n2", "avian influenza a(h5n5)"):
            assert _resolve_pathogen_key(subtype, respiratory) != "h5n1", subtype
