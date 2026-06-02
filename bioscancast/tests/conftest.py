"""Pytest configuration shared across the bioscancast test suite.

Adds a `--live` CLI flag that opts into tests marked `@pytest.mark.live`,
which require network access to real third-party services (e.g. CDN-fronted
sources used to verify the curl_cffi fetcher against Cloudflare-style
TLS-fingerprint blocks). By default these tests are skipped so the suite
stays hermetic in CI.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.live (hits real network endpoints).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: test requires real network access; opt in with --live.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="needs --live to run (hits real network)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
