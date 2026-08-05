"""Tests for the always-on recommendation server wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _server_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "homelab"
        / "server.py"
    )
    spec = importlib.util.spec_from_file_location("homelab_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _request(module, path):
    request = module.Handler.__new__(module.Handler)
    request.path = path
    sent = []
    request._send = lambda code, body, cacheable=True: sent.append(
        (code, body, cacheable)
    )
    return request, sent


def test_health_endpoint_does_not_load_or_cache_recommendations():
    module = _server_module()
    request, sent = _request(module, "/healthz")

    request.do_GET()

    assert sent == [(200, {"ok": True}, False)]


def test_recommendation_endpoint_delegates_to_shared_handler(monkeypatch):
    module = _server_module()
    request, sent = _request(
        module,
        "/api/spicetify_recommend?query=Redbone+%E2%80%94+Childish+Gambino",
    )
    delegated = []
    monkeypatch.setattr(
        module.RecommendationHandler,
        "do_GET",
        lambda self: delegated.append(self.path),
    )

    request.do_GET()

    assert delegated == [request.path]
    assert sent == []


def test_unknown_endpoint_returns_uncached_404():
    module = _server_module()
    request, sent = _request(module, "/api/unknown")

    request.do_GET()

    assert sent == [(404, {"ok": False, "error": "not found"}, False)]
