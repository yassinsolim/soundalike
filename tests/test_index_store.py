"""Tests for the deep-vibe pack store (manifest + resolution + download).

Network is avoided by pointing the "release" at a local file:// URL and by
exercising the bundled/cached resolution branches directly.
"""

from __future__ import annotations

import hashlib
import json

from soundalike.ml import index_store


def _sha(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_manifest_loads_and_describes():
    # The repo ships a manifest; it should parse and describe cleanly.
    m = index_store.load_manifest()
    assert m is not None
    assert "index" in m and "encoder" in m
    assert m["index"]["n_tracks"] > 0
    text = index_store.describe()
    assert "tracks" in text


def test_dual_sonic64_manifest_shape():
    manifest = index_store.load_manifest()
    assert manifest["release_tag"] == "index-2026.07.11-dual-sonic64"
    assert manifest["index"] == {
        "asset": "deepvibe_index.npz",
        "sha256": "f3ed57af1b8073f2872eed1e9192dee04d1089c7266fb98a157d1ea194526fb9",
        "n_tracks": 272853,
        "bytes": 299288526,
        "sonic_dim": 64,
        "clap_dim": 64,
        "source_prior_columns": 2,
    }


def test_resolve_prefers_matching_cache(tmp_path, monkeypatch):
    # A cached file whose sha matches the manifest spec is used without download.
    asset = tmp_path / "thing.bin"
    asset.write_bytes(b"hello world" * 100)
    spec = {"asset": "thing.bin", "sha256": _sha(asset)}
    manifest = {"repo": "x/y", "release_tag": "t"}

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "thing.bin").write_bytes(asset.read_bytes())

    calls = []
    monkeypatch.setattr(index_store, "_download",
                        lambda *a, **k: calls.append(a) or False)
    got = index_store._resolve_one(spec, manifest, cache, force=False, progress=lambda m: None)
    assert got == cache / "thing.bin"
    assert calls == []  # never attempted a download


def test_resolve_downloads_when_missing(tmp_path, monkeypatch):
    spec = {"asset": "thing.bin", "sha256": "deadbeef"}
    manifest = {"repo": "x/y", "release_tag": "t"}
    cache = tmp_path / "cache"

    def fake_download(url, dest, sha, progress):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded")
        return True

    monkeypatch.setattr(index_store, "_download", fake_download)
    monkeypatch.setattr(index_store, "_bundled", lambda name: None)
    got = index_store._resolve_one(spec, manifest, cache, force=False, progress=lambda m: None)
    assert got == cache / "thing.bin"
    assert got.read_bytes() == b"downloaded"


def test_resolve_falls_back_to_bundled_on_failure(tmp_path, monkeypatch):
    spec = {"asset": "thing.bin", "sha256": "deadbeef"}
    manifest = {"repo": "x/y", "release_tag": "t"}
    cache = tmp_path / "cache"
    bundled = tmp_path / "bundled_thing.bin"
    bundled.write_bytes(b"bundled copy")

    monkeypatch.setattr(index_store, "_download", lambda *a, **k: False)
    # First bundled() call (pre-download, sha check) fails; fallback returns it.
    monkeypatch.setattr(index_store, "_bundled",
                        lambda name: bundled if name == "thing.bin" else None)
    got = index_store._resolve_one(spec, manifest, cache, force=False, progress=lambda m: None)
    assert got == bundled  # graceful offline fallback


def test_asset_url_construction():
    manifest = {"repo": "owner/repo", "release_tag": "index-v1"}
    url = index_store._asset_url(manifest, "deepvibe_index.npz")
    assert url == "https://github.com/owner/repo/releases/download/index-v1/deepvibe_index.npz"
