from bioscancast.stages.search_stage.url_normalization import extract_domain, normalize_url


class TestNormalizeUrl:
    def test_strips_trailing_slash(self):
        assert normalize_url("https://example.com/path/") == "https://example.com/path"

    def test_strips_fragment(self):
        assert normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_strips_tracking_params(self):
        url = "https://example.com/page?utm_source=twitter&utm_medium=social&id=42"
        result = normalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "id=42" in result

    def test_strips_fbclid(self):
        url = "https://example.com/page?fbclid=abc123&topic=health"
        result = normalize_url(url)
        assert "fbclid" not in result
        assert "topic=health" in result

    def test_preserves_content_query_params(self):
        url = "https://example.com/search?q=h5n1+cases&lang=en"
        result = normalize_url(url)
        assert "q=" in result
        assert "lang=en" in result

    def test_lowercases_scheme_and_host(self):
        result = normalize_url("HTTPS://WWW.Example.COM/Path")
        assert result.startswith("https://example.com")

    def test_strips_www_prefix(self):
        result = normalize_url("https://www.who.int/page")
        assert "www." not in result
        assert "who.int" in result

    def test_strips_default_port(self):
        assert normalize_url("https://example.com:443/page") == "https://example.com/page"
        assert normalize_url("http://example.com:80/page") == "http://example.com/page"

    def test_preserves_non_default_port(self):
        result = normalize_url("https://example.com:8080/page")
        assert ":8080" in result

    def test_idempotent(self):
        url = "https://example.com/page?utm_source=x&id=1#frag"
        once = normalize_url(url)
        twice = normalize_url(once)
        assert once == twice

    def test_empty_path(self):
        result = normalize_url("https://example.com")
        assert result == "https://example.com"


class TestExtractDomain:
    def test_basic(self):
        assert extract_domain("https://www.cdc.gov/page") == "cdc.gov"

    def test_subdomain(self):
        assert extract_domain("https://wwwnc.cdc.gov/page") == "wwwnc.cdc.gov"

    def test_no_www(self):
        assert extract_domain("https://nature.com/articles/123") == "nature.com"

    def test_lowercases(self):
        assert extract_domain("https://WHO.INT/page") == "who.int"
