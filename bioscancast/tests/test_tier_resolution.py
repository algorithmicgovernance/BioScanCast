from bioscancast.stages.search_stage.tier_resolution import (
    is_aggregator_domain,
    is_official_domain,
    resolve_tier,
)


class TestResolveTier:
    def test_tier1_health_authority(self):
        tier, score, label = resolve_tier("who.int")
        assert tier == 1
        assert score == 1.0
        assert label == "official"

    def test_tier1_journal(self):
        tier, score, label = resolve_tier("nature.com")
        assert tier == 1
        assert score == 1.0
        assert label == "academic"

    def test_tier2_government(self):
        tier, score, label = resolve_tier("nih.gov")
        assert tier == 2
        assert score == 0.8
        assert label == "official"

    def test_tier2_preprint(self):
        tier, score, label = resolve_tier("medrxiv.org")
        assert tier == 2
        assert score == 0.8
        assert label == "academic"

    def test_tier2_wire(self):
        tier, score, label = resolve_tier("reuters.com")
        assert tier == 2
        assert score == 0.8
        assert label == "trusted_media"

    def test_tier3_media(self):
        tier, score, label = resolve_tier("bbc.com")
        assert tier == 3
        assert score == 0.6
        assert label == "trusted_media"

    def test_tier4_ngo(self):
        tier, score, label = resolve_tier("msf.org")
        assert tier == 4
        assert score == 0.4
        assert label == "ngo"

    def test_unknown_domain(self):
        tier, score, label = resolve_tier("randomblog.xyz")
        assert tier == 5
        assert score == 0.2
        assert label == "unknown"

    def test_subdomain_match(self):
        """wwwnc.cdc.gov should match cdc.gov via second-level domain."""
        tier, score, label = resolve_tier("wwwnc.cdc.gov")
        assert tier == 1
        assert label == "official"

    def test_www_subdomain(self):
        tier, score, label = resolve_tier("www.who.int")
        assert tier == 1

    def test_tld_heuristic_dot_gov(self):
        """Unknown .gov domain → Tier 3 (0.6) to avoid over-crediting."""
        tier, score, label = resolve_tier("randomagency.gov")
        assert tier == 3
        assert score == 0.6
        assert label == "official"

    def test_tld_heuristic_dot_edu(self):
        tier, score, label = resolve_tier("someuniversity.edu")
        assert tier == 3
        assert score == 0.6
        assert label == "academic"

    def test_tld_heuristic_dot_int(self):
        tier, score, label = resolve_tier("someorg.int")
        assert tier == 2
        assert score == 0.8
        assert label == "official"

    def test_case_insensitive(self):
        tier, _, label = resolve_tier("WHO.INT")
        assert tier == 1
        assert label == "official"


class TestIsOfficialDomain:
    def test_who_is_official(self):
        assert is_official_domain("who.int") is True

    def test_cdc_is_official(self):
        assert is_official_domain("cdc.gov") is True

    def test_journal_not_official(self):
        assert is_official_domain("nature.com") is False

    def test_unknown_not_official(self):
        assert is_official_domain("random.com") is False


class TestIsAggregatorDomain:
    def test_metaculus(self):
        assert is_aggregator_domain("metaculus.com") is True

    def test_polymarket(self):
        assert is_aggregator_domain("polymarket.com") is True

    def test_subdomain_of_aggregator(self):
        assert is_aggregator_domain("www.metaculus.com") is True

    def test_non_aggregator(self):
        assert is_aggregator_domain("cdc.gov") is False
