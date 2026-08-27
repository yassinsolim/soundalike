"""Tests for the broad-harvest helpers (data structures only, no network)."""

from __future__ import annotations

import json

import pytest

from soundalike.audio.previews import DeezerTrack
from soundalike.ml.grow_broad import (
    ApiCallBudgetExceeded,
    BROAD_SEED_ARTISTS,
    _BudgetedSession,
    _fresh_preview,
    _load_candidates,
    _save_candidates,
    harvest_targeted_to_cache,
)


def test_seed_list_spans_many_scenes():
    # A broad list is the whole point — guard against it shrinking back to one
    # scene. Broad + niche together should be large and unique-ish.
    from soundalike.ml.grow_broad import BROAD_SEED_ARTISTS, NICHE_SEED_ARTISTS

    combined = BROAD_SEED_ARTISTS + NICHE_SEED_ARTISTS
    assert len(combined) >= 300
    # A few duplicate names across the broad/niche lists are harmless — the crawl
    # dedups by resolved artist id — so only guard against gross duplication.
    assert len(set(combined)) >= len(combined) * 0.9


def test_candidates_roundtrip(tmp_path):
    tracks = [
        DeezerTrack(id=1, title="A", artist="X", artist_id=10, preview_url="http://p/1.mp3"),
        DeezerTrack(id=2, title="B", artist="Y", artist_id=20, preview_url="http://p/2.mp3"),
    ]
    p = tmp_path / "cands.json"
    _save_candidates(p, tracks)
    back = _load_candidates(p)
    assert [t.id for t in back] == [1, 2]
    assert [t.title for t in back] == ["A", "B"]
    assert [t.artist for t in back] == ["X", "Y"]
    assert all(t.preview_url for t in back)


def test_candidates_roundtrip_missing_preview(tmp_path):
    # A row without a preview should still load (empty url), not crash.
    tracks = [DeezerTrack(id=5, title="T", artist="Z", artist_id=0, preview_url="")]
    p = tmp_path / "c.json"
    _save_candidates(p, tracks)
    back = _load_candidates(p)
    assert len(back) == 1 and back[0].preview_url == ""


def _targeted_report(path, plan):
    path.write_text(json.dumps({"targeted_crawl_plan": plan}), encoding="utf-8")
    return path


def test_targeted_dry_run_uses_plan_without_network_clients(tmp_path, monkeypatch):
    report = _targeted_report(tmp_path / "audit.json", [{
        "artist": "Missing Artist", "category": "proxy", "observed": 0,
        "minimum": 2, "reason": "missing",
    }])

    def fail_if_network_client_is_created(*_args, **_kwargs):
        raise AssertionError("dry run must not construct a network client")

    monkeypatch.setattr("soundalike.ml.grow_broad._gather_artist_ids", fail_if_network_client_is_created)
    monkeypatch.setattr("soundalike.ml.grow_broad.DeezerClient", fail_if_network_client_is_created)
    monkeypatch.setattr("soundalike.ml.grow_broad.requests.Session", fail_if_network_client_is_created)
    assert harvest_targeted_to_cache(
        tmp_path / "cache.npz", report, max_artists=1, max_tracks=2,
        max_api_calls=4, dry_run=True,
    ) is None


def test_targeted_crawl_refuses_unbounded_or_insufficient_api_budgets(tmp_path):
    report = _targeted_report(tmp_path / "audit.json", [{
        "artist": "Missing Artist", "category": "proxy", "observed": 0,
        "minimum": 2, "reason": "missing",
    }])

    with pytest.raises(ValueError, match="finite positive budgets"):
        harvest_targeted_to_cache(tmp_path / "cache.npz", report, max_artists=1, max_tracks=2)
    with pytest.raises(ValueError, match="exceeds max_api_calls=3"):
        harvest_targeted_to_cache(
            tmp_path / "cache.npz", report, max_artists=1, max_tracks=2, max_api_calls=3,
        )


def test_targeted_api_budget_is_enforced_across_retries(monkeypatch):
    class Response:
        @staticmethod
        def json():
            return {"error": {"code": 4}}

    class Session:
        calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

    raw = Session()
    budgeted = _BudgetedSession(raw, max_calls=1)
    monkeypatch.setattr("soundalike.ml.grow_broad.time.sleep", lambda _seconds: None)

    with pytest.raises(ApiCallBudgetExceeded, match="max_api_calls=1"):
        _fresh_preview(123, budgeted)

    assert raw.calls == 1
    assert budgeted.used == 1
