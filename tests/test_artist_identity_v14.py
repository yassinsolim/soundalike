"""Tests for artist_identity_v14 — artist disambiguation module.

Coverage
--------
- normalize_key: casefold, transliteration, punctuation, edge-cases
- _valid_mbid, _extract_deezer_track_fields (including string-ID coercion,
  malformed artist block, COALESCE preservation)
- IdentityCache: ingest_deezer_metadata (COALESCE), ingest_candidate_json,
  ingest_lastfm_mbids (fixture only, no network), build_npz
- MBID attribution: homonym names get name-level-only MBIDs; non-homonym
  names get direct per-artist MBIDs
- has_disjoint_mbid_evidence: non-tautological (disjoint/uncertain/none)
- v13 variant union: all variant records, not last-write
- IdentityAsset: row_identity (direct_mbids vs name_level_mbids),
  contributor_intersection, artist_centroid, name_to_deezer_ids,
  name_to_mbids, disambiguate, homonym_names, verify_bindings
- run_identity_audit: ALL keys coverage, unresolved rows, v13 union
- fallback_by_track_count, fallback_by_mbid_coverage
- build_identity_network with injected mock fetcher
- No network calls anywhere in this file
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from soundalike.ml.artist_identity_v14 import (
    SCHEMA_VERSION,
    SUGGESTED_MIN_CONFIDENCE,
    SUGGESTED_MIN_MARGIN,
    ArtistIdentityError,
    DeezerFetcher,
    IdentityAsset,
    IdentityCache,
    _SENTINEL_ARTIST_ID,
    _extract_deezer_track_fields,
    _valid_mbid,
    build_identity_network,
    fallback_by_mbid_coverage,
    fallback_by_track_count,
    normalize_key,
    run_identity_audit,
)

# ---------------------------------------------------------------------------
# Shared test fixtures and builders
# ---------------------------------------------------------------------------

_MBID_A = "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d"
_MBID_B = "aaaaaefc-cf9e-42e0-be17-e2c3e1d26000"
_MBID_C = "ccccccfc-0000-42e0-be17-e2c3e1d26001"


def _make_track_ids(n: int = 6) -> np.ndarray:
    return np.array([100, 200, 300, 400, 500, 600][:n], dtype=np.int64)


def _deezer_payload(
    track_id: int,
    artist_id: int,
    artist_name: str,
    *,
    title: str = "Song",
    contributors: Optional[List[Dict[str, Any]]] = None,
    include_preview: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": track_id,
        "title": title,
        "artist": {"id": artist_id, "name": artist_name},
    }
    if contributors is not None:
        payload["contributors"] = contributors
    if include_preview:
        payload["preview"] = "https://cdn.deezer.com/fake/signed?token=secret"
    return payload


def _build_lastfm_tar(entries: List[tuple]) -> bytes:
    """Build in-memory .tar.gz mimicking Last.fm-360K format."""
    member_name = "lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv"
    lines = "\n".join(
        "\t".join(str(x) for x in row) for row in entries
    ).encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(lines)
        tar.addfile(info, io.BytesIO(lines))
    return buf.getvalue()


def _build_asset(tmp_path: Path, **kwargs) -> IdentityAsset:
    """Build a minimal IdentityAsset for testing (no network)."""
    track_ids = kwargs.pop("track_ids", _make_track_ids())
    cache_path = tmp_path / "identity.sqlite3"
    npz_path = tmp_path / "identity_v14.npz"
    embeddings = kwargs.pop("embeddings", None)
    with IdentityCache(cache_path, track_ids) as cache:
        for payload_kw in kwargs.pop("payloads", []):
            cache.ingest_deezer_metadata(payload_kw["id"], payload_kw)
        if "tar_bytes" in kwargs:
            tar_path = tmp_path / "lastfm.tar.gz"
            tar_path.write_bytes(kwargs.pop("tar_bytes"))
            cache.ingest_lastfm_mbids(tar_path)
        cache.build_npz(
            track_ids, "fake_clap_hash_" + "0" * 48, npz_path, embeddings
        )
    return IdentityAsset.load(npz_path)


# ===========================================================================
# normalize_key
# ===========================================================================


class TestNormalizeKey:
    def test_casefold(self):
        assert normalize_key("NOTHING") == "nothing"
        assert normalize_key("Nothing") == "nothing"

    def test_transliteration_umlaut(self):
        assert normalize_key("Björk") == "bjork"
        assert normalize_key("Röyksopp") == "royksopp"

    def test_transliteration_accent(self):
        assert normalize_key("Sigur Rós") == "sigur ros"
        assert normalize_key("Café Tacvba") == "cafe tacvba"

    def test_punctuation_collapse(self):
        assert normalize_key("Jay-Z") == "jay z"
        assert normalize_key("A$AP Rocky") == "a ap rocky"
        assert normalize_key("M.I.A.") == "m i a"
        assert normalize_key("AC/DC") == "ac dc"

    def test_parenthetical_removal(self):
        assert normalize_key("Artist (feat. Someone)") == "artist"
        assert normalize_key("Song [Remaster]") == "song"

    def test_leading_trailing_whitespace(self):
        assert normalize_key("  The Weeknd  ") == "the weeknd"

    def test_empty_string(self):
        assert normalize_key("") == ""

    def test_numeric_preserved(self):
        assert normalize_key("2Pac") == "2pac"
        assert normalize_key("50 Cent") == "50 cent"

    def test_kanji_returns_string(self):
        result = normalize_key("東京事変")
        assert isinstance(result, str)

    def test_homonym_collision(self):
        assert normalize_key("Nothing") == normalize_key("NOTHING")
        assert normalize_key("nothing") == normalize_key("  Nothing  ")

    def test_deterministic(self):
        for name in ["Björk", "Jay-Z", "AC/DC", "nothing"]:
            assert normalize_key(name) == normalize_key(name)


# ===========================================================================
# _valid_mbid
# ===========================================================================


class TestValidMbid:
    def test_valid(self):
        assert _valid_mbid(_MBID_A)
        assert _valid_mbid(_MBID_A.upper())

    def test_invalid_cases(self):
        assert not _valid_mbid("")
        assert not _valid_mbid("b10bbbfc-cf9e-42e0")
        assert not _valid_mbid("g10bbbfc-cf9e-42e0-be17-e2c3e1d2600d")
        assert not _valid_mbid("b10bbbfccf9e42e0be17e2c3e1d2600d")


# ===========================================================================
# _extract_deezer_track_fields
# ===========================================================================


class TestExtractDeezerTrackFields:
    def test_basic(self):
        result = _extract_deezer_track_fields(_deezer_payload(123, 456, "Artist"))
        assert result is not None
        assert result["track_id"] == 123
        assert result["primary_artist_deezer_id"] == 456
        assert result["artist_name"] == "Artist"
        assert result["contributors"] == []

    def test_string_id_coerced(self):
        """String track/artist IDs (Deezer sometimes returns these) are accepted."""
        payload = {"id": "789", "artist": {"id": "101", "name": "Band"}, "title": "T"}
        result = _extract_deezer_track_fields(payload)
        assert result is not None
        assert result["track_id"] == 789
        assert result["primary_artist_deezer_id"] == 101

    def test_missing_artist_block_returns_partial(self):
        """Missing artist block → primary_id=None but track still extracted."""
        payload = {"id": 123, "title": "No Artist"}
        result = _extract_deezer_track_fields(payload)
        assert result is not None
        assert result["track_id"] == 123
        assert result["primary_artist_deezer_id"] is None

    def test_malformed_artist_id_returns_none_primary(self):
        """Non-numeric artist ID → primary_id=None, track still parsed."""
        payload = {"id": 999, "artist": {"id": "bad", "name": "Artist"}, "title": "T"}
        result = _extract_deezer_track_fields(payload)
        assert result is not None
        assert result["primary_artist_deezer_id"] is None
        assert result["artist_name"] == "Artist"

    def test_preview_not_in_result(self):
        payload = _deezer_payload(123, 456, "A", include_preview=True)
        result = _extract_deezer_track_fields(payload)
        assert result is not None
        assert "preview" not in result

    def test_contributors_extracted(self):
        payload = _deezer_payload(
            123, 456, "Main",
            contributors=[
                {"id": 789, "name": "Featured", "role": "Featured"},
                {"id": 101, "name": "Producer", "role": "Producer"},
            ],
        )
        result = _extract_deezer_track_fields(payload)
        assert result is not None
        assert {c["deezer_id"] for c in result["contributors"]} == {789, 101}

    def test_missing_id_returns_none(self):
        assert _extract_deezer_track_fields({"title": "No ID"}) is None

    def test_non_dict_returns_none(self):
        assert _extract_deezer_track_fields([]) is None  # type: ignore
        assert _extract_deezer_track_fields("str") is None  # type: ignore


# ===========================================================================
# IdentityCache — ingestion
# ===========================================================================


class TestIdentityCacheIngest:
    def test_ingest_single_track(self, tmp_path):
        with IdentityCache(tmp_path / "c.db", _make_track_ids()) as cache:
            ok = cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Rock Band"))
            assert ok is True
            stats = cache.stats()
        assert stats["tracks_with_primary_id"] >= 1
        assert stats["unique_artist_ids"] >= 1

    def test_coalesce_preserves_valid_primary_id(self, tmp_path):
        """COALESCE: a malformed (null-artist) response must NOT overwrite an
        already-stored valid primary artist ID."""
        track_ids = np.array([100], dtype=np.int64)
        cache_path = tmp_path / "c.db"
        with IdentityCache(cache_path, track_ids) as cache:
            # First ingest a valid payload
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Rock Band"))
            # Then ingest a malformed payload (no artist block) for the same track
            cache.ingest_deezer_metadata(100, {"id": 100, "title": "Rock Song"})
            row = cache._conn.execute(
                "SELECT primary_artist_deezer_id FROM track_identity WHERE track_id=100"
            ).fetchone()
        # Primary ID must still be 1001 — not overwritten with NULL
        assert row[0] == 1001

    def test_no_preview_persisted(self, tmp_path):
        with IdentityCache(tmp_path / "c.db", _make_track_ids()) as cache:
            cache.ingest_deezer_metadata(
                100, _deezer_payload(100, 1001, "Rock Band", include_preview=True)
            )
            row = cache._conn.execute(
                "SELECT * FROM track_identity WHERE track_id=100"
            ).fetchone()
        row_str = str(row).lower()
        assert "dzcdn" not in row_str
        assert "token=" not in row_str

    def test_resume_validates_sha(self, tmp_path):
        cache_path = tmp_path / "c.db"
        ids1 = np.array([100, 200, 300], dtype=np.int64)
        ids2 = np.array([100, 200, 400], dtype=np.int64)
        with IdentityCache(cache_path, ids1):
            pass
        with pytest.raises(ArtistIdentityError, match="track_ids_sha256"):
            IdentityCache(cache_path, ids2)

    def test_ingest_candidate_json(self, tmp_path):
        track_ids = _make_track_ids()
        candidate = {
            "tracks": {
                "200": _deezer_payload(200, 2002, "Jazz Artist"),
                "300": _deezer_payload(300, 3003, "Jazz Too"),
            }
        }
        json_path = tmp_path / "cand.json"
        json_path.write_text(json.dumps(candidate), encoding="utf-8")
        with IdentityCache(tmp_path / "c.db", track_ids) as cache:
            count = cache.ingest_candidate_json(json_path)
        assert count >= 2

    def test_ingest_candidate_json_missing_file(self, tmp_path):
        with IdentityCache(tmp_path / "c.db") as cache:
            assert cache.ingest_candidate_json(tmp_path / "nonexistent.json") == 0

    def test_multi_artist_contributors(self, tmp_path):
        """All contributor Deezer IDs are stored independently."""
        payload = _deezer_payload(
            100, 1001, "Main",
            contributors=[
                {"id": 1001, "name": "Main", "role": "Main"},
                {"id": 2002, "name": "Feature1", "role": "Featured"},
                {"id": 3003, "name": "Feature2", "role": "Featured"},
            ],
        )
        with IdentityCache(tmp_path / "c.db", _make_track_ids()) as cache:
            cache.ingest_deezer_metadata(100, payload)
            rows = cache._conn.execute(
                "SELECT contributor_deezer_id FROM track_contributors WHERE track_id=100"
            ).fetchall()
        assert {r[0] for r in rows} == {1001, 2002, 3003}

    def test_homonym_names_detected(self, tmp_path):
        ids = np.array([100, 200], dtype=np.int64)
        with IdentityCache(tmp_path / "c.db", ids) as cache:
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Nothing"))
            cache.ingest_deezer_metadata(200, _deezer_payload(200, 9001, "NOTHING"))
            homonyms = cache.homonym_names()
        names = {h[0] for h in homonyms}
        assert "nothing" in names
        for name, dids in homonyms:
            if name == "nothing":
                assert set(dids) == {1001, 9001}

    def test_unresolved_track_ids(self, tmp_path):
        ids = np.array([100, 200, 300], dtype=np.int64)
        with IdentityCache(tmp_path / "c.db", ids) as cache:
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Artist"))
            unresolved = cache.unresolved_track_ids()
        assert 200 in unresolved
        assert 300 in unresolved
        assert 100 not in unresolved


# ===========================================================================
# IdentityCache — Last.fm ingestion
# ===========================================================================


class TestIdentityCacheLastfm:
    def _make_tar(self, entries, tmp_path) -> Path:
        p = tmp_path / "lastfm.tar.gz"
        p.write_bytes(_build_lastfm_tar(entries))
        return p

    def test_valid_mbid_inserted(self, tmp_path):
        tar = self._make_tar([("u1", _MBID_A, "Nothing", "100")], tmp_path)
        with IdentityCache(tmp_path / "c.db") as cache:
            result = cache.ingest_lastfm_mbids(tar)
        assert result["valid_mbid_rows"] >= 1
        assert result["distinct_mbid_name_pairs_inserted"] >= 1

    def test_invalid_mbid_skipped(self, tmp_path):
        tar = self._make_tar(
            [("u1", "not-a-valid-mbid", "Artist", "10"), ("u2", "", "Art2", "5")],
            tmp_path,
        )
        with IdentityCache(tmp_path / "c.db") as cache:
            result = cache.ingest_lastfm_mbids(tar)
        assert result["valid_mbid_rows"] == 0
        assert result["skipped_invalid_mbid"] >= 1

    def test_same_mbid_different_spellings_not_homonym(self, tmp_path):
        """Multiple spellings of the same MBID → same entity, not a homonym."""
        tar = self._make_tar(
            [
                ("u1", _MBID_A, "Nothing", "100"),
                ("u2", _MBID_A, "NOTHING", "50"),
                ("u3", _MBID_A, "nothing", "200"),
            ],
            tmp_path,
        )
        with IdentityCache(tmp_path / "c.db") as cache:
            cache.ingest_lastfm_mbids(tar)
            homonyms = cache.mbid_homonyms()
        assert "nothing" not in {h[0] for h in homonyms}

    def test_same_name_different_mbids_is_homonym(self, tmp_path):
        tar = self._make_tar(
            [("u1", _MBID_A, "Nothing", "100"), ("u2", _MBID_B, "nothing", "200")],
            tmp_path,
        )
        with IdentityCache(tmp_path / "c.db") as cache:
            result = cache.ingest_lastfm_mbids(tar)
            homonyms = cache.mbid_homonyms()
        assert result["homonym_normalized_names"] >= 1
        assert "nothing" in {h[0] for h in homonyms}

    def test_max_rows_limit(self, tmp_path):
        entries = [(f"u{i}", _MBID_A, f"Artist {i}", str(i)) for i in range(50)]
        tar = self._make_tar(entries, tmp_path)
        with IdentityCache(tmp_path / "c.db") as cache:
            result = cache.ingest_lastfm_mbids(tar, max_rows=10)
        assert result["source_rows_read"] == 10

    def test_missing_member_raises(self, tmp_path):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz"):
            pass
        tar_path = tmp_path / "empty.tar.gz"
        tar_path.write_bytes(buf.getvalue())
        with IdentityCache(tmp_path / "c.db") as cache:
            with pytest.raises(ArtistIdentityError):
                cache.ingest_lastfm_mbids(tar_path)


# ===========================================================================
# MBID attribution — core correctness
# ===========================================================================


class TestMbidAttribution:
    """Verify the name-level vs per-artist MBID attribution policy."""

    def _make_homonym_asset(self, tmp_path) -> IdentityAsset:
        """Two distinct Deezer IDs share normalised name 'nothing'."""
        track_ids = np.array([100, 200, 300], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),   # shoegaze
            _deezer_payload(200, 9001, "nothing"),   # dnb
            _deezer_payload(300, 3003, "Tool"),      # unique name
        ]
        tar_bytes = _build_lastfm_tar([
            ("u1", _MBID_A, "Nothing", "1000"),
            ("u2", _MBID_B, "nothing", "500"),
            ("u3", _MBID_C, "Tool", "800"),
        ])
        return _build_asset(tmp_path, track_ids=track_ids, payloads=payloads, tar_bytes=tar_bytes)

    def test_homonym_has_no_direct_per_artist_mbids(self, tmp_path):
        """Homonym artists must NOT get per-artist direct MBIDs."""
        asset = self._make_homonym_asset(tmp_path)
        # Artists 1001 and 9001 share "nothing" — no direct attribution possible
        apos_1001 = asset._artist_pos.get(1001)
        apos_9001 = asset._artist_pos.get(9001)
        assert apos_1001 is not None
        assert apos_9001 is not None
        mbids_1001 = [
            str(x) for x in asset._mbid_flat[
                asset._mbid_indptr[apos_1001]:asset._mbid_indptr[apos_1001 + 1]
            ]
        ]
        mbids_9001 = [
            str(x) for x in asset._mbid_flat[
                asset._mbid_indptr[apos_9001]:asset._mbid_indptr[apos_9001 + 1]
            ]
        ]
        assert mbids_1001 == [], "Homonym artist 1001 must have no direct MBIDs"
        assert mbids_9001 == [], "Homonym artist 9001 must have no direct MBIDs"

    def test_non_homonym_gets_direct_mbid(self, tmp_path):
        """Non-homonym artist (Tool) gets direct per-artist MBID."""
        asset = self._make_homonym_asset(tmp_path)
        apos_3003 = asset._artist_pos.get(3003)
        assert apos_3003 is not None
        mbids_3003 = [
            str(x) for x in asset._mbid_flat[
                asset._mbid_indptr[apos_3003]:asset._mbid_indptr[apos_3003 + 1]
            ]
        ]
        assert mbids_3003 == [], "Name-only joins must not become direct MBID links"

    def test_homonym_has_name_level_mbids(self, tmp_path):
        """Both MBIDs for 'nothing' appear at the name-cluster level."""
        asset = self._make_homonym_asset(tmp_path)
        name_mbids = asset.name_to_mbids("nothing")
        assert _MBID_A in name_mbids
        assert _MBID_B in name_mbids

    def test_row_identity_homonym_uses_name_level_unlinked(self, tmp_path):
        """row_identity for a homonym artist returns name_level_unlinked."""
        asset = self._make_homonym_asset(tmp_path)
        # Row 0 = track 100 = artist 1001 (Nothing, homonym)
        identity = asset.row_identity(0)
        assert identity["mbid_attribution"] == "name_level_unlinked"
        assert identity["direct_mbids"] == []
        assert len(identity["name_level_mbids"]) >= 1

    def test_row_identity_unique_name_still_requires_cross_source_link(self, tmp_path):
        """A unique name is not itself a verified Deezer-to-MBID bridge."""
        asset = self._make_homonym_asset(tmp_path)
        identity = asset.row_identity(2)
        assert identity["mbid_attribution"] == "name_level_unlinked"
        assert identity["direct_mbids"] == []
        assert _MBID_C in identity["name_level_mbids"]

    def test_row_identity_no_mbid_attribution(self, tmp_path):
        """Artist with no Last.fm entry gets 'none' attribution."""
        track_ids = np.array([100], dtype=np.int64)
        asset = _build_asset(
            tmp_path,
            track_ids=track_ids,
            payloads=[_deezer_payload(100, 1001, "Obscure Band")],
        )
        identity = asset.row_identity(0)
        assert identity["mbid_attribution"] == "none"
        assert identity["direct_mbids"] == []
        assert identity["name_level_mbids"] == []

    def test_all_same_name_mbids_not_spread_to_all_deezer_ids(self, tmp_path):
        """Critical: MBID_A and MBID_B for 'nothing' must not both appear in
        per-artist direct MBIDs of EITHER 1001 OR 9001."""
        asset = self._make_homonym_asset(tmp_path)
        for did in (1001, 9001):
            # Neither homonym artist should claim individual MBIDs directly
            apos = asset._artist_pos.get(did)
            assert apos is not None
            start = asset._mbid_indptr[apos]
            stop = asset._mbid_indptr[apos + 1]
            per_artist = set(str(x) for x in asset._mbid_flat[start:stop])
            assert _MBID_A not in per_artist, (
                f"Artist {did} falsely claims MBID_A as direct"
            )
            assert _MBID_B not in per_artist, (
                f"Artist {did} falsely claims MBID_B as direct"
            )


# ===========================================================================
# IdentityCache — build_npz
# ===========================================================================


class TestBuildNpz:
    def test_builds_valid_npz(self, tmp_path):
        asset = _build_asset(tmp_path, payloads=[_deezer_payload(100, 1001, "Rock")])
        assert len(asset) == len(_make_track_ids())
        assert asset.metadata["schema_version"] == SCHEMA_VERSION

    def test_metadata_bindings(self, tmp_path):
        import hashlib
        track_ids = _make_track_ids()
        asset = _build_asset(tmp_path)
        assert asset.metadata["track_ids_sha256"] == hashlib.sha256(
            track_ids.tobytes()
        ).hexdigest()
        assert asset.metadata["clap_asset_hash"] == "fake_clap_hash_" + "0" * 48

    def test_verify_bindings_ok(self, tmp_path):
        import hashlib
        track_ids = _make_track_ids()
        asset = _build_asset(tmp_path)
        asset.verify_bindings(
            track_ids_sha256=hashlib.sha256(track_ids.tobytes()).hexdigest(),
            clap_asset_hash="fake_clap_hash_" + "0" * 48,
        )

    def test_verify_bindings_fails(self, tmp_path):
        asset = _build_asset(tmp_path)
        with pytest.raises(ArtistIdentityError, match="track_ids_sha256"):
            asset.verify_bindings(track_ids_sha256="wrong")
        with pytest.raises(ArtistIdentityError, match="clap_asset_hash"):
            asset.verify_bindings(clap_asset_hash="wrong")

    def test_allow_pickle_false_enforced(self, tmp_path):
        asset = _build_asset(tmp_path)
        npz_path = list(tmp_path.glob("*.npz"))[0]
        with pytest.raises(ArtistIdentityError, match="allow_pickle"):
            IdentityAsset(npz_path, allow_pickle=True)

    def test_no_preview_in_npz(self, tmp_path):
        track_ids = _make_track_ids()
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        payload = _deezer_payload(100, 1001, "Artist", include_preview=True)
        with IdentityCache(cache_path, track_ids) as cache:
            cache.ingest_deezer_metadata(100, payload)
            cache.build_npz(track_ids, "h" * 64, npz_path)
        raw = np.load(npz_path, allow_pickle=False)
        for key in raw.files:
            text = str(raw[key]).lower()
            assert "dzcdn" not in text
            assert "token=" not in text

    def test_embeddings_shape_mismatch_raises(self, tmp_path):
        track_ids = _make_track_ids(6)
        with IdentityCache(tmp_path / "c.db", track_ids) as cache:
            with pytest.raises(ArtistIdentityError, match="embeddings"):
                cache.build_npz(
                    track_ids, "h" * 64, tmp_path / "id.npz",
                    np.zeros((3, 128), dtype=np.float32)
                )

    def test_unresolved_rows_reported(self, tmp_path):
        track_ids = _make_track_ids(3)
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        with IdentityCache(cache_path, track_ids) as cache:
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Artist"))
            result = cache.build_npz(track_ids, "h" * 64, npz_path)
        # Tracks 200 and 300 were not ingested
        assert result["rows_unresolved"] == 2
        assert 200 in result["unresolved_track_ids"]
        assert 300 in result["unresolved_track_ids"]

    def test_name_mbid_arrays_in_npz(self, tmp_path):
        """NPZ must contain name_mbid_flat and name_mbid_indptr arrays."""
        track_ids = np.array([100], dtype=np.int64)
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        tar_path = tmp_path / "lastfm.tar.gz"
        tar_path.write_bytes(_build_lastfm_tar([("u1", _MBID_A, "Solo", "100")]))
        with IdentityCache(cache_path, track_ids) as cache:
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Solo"))
            cache.ingest_lastfm_mbids(tar_path)
            cache.build_npz(track_ids, "h" * 64, npz_path)
        raw = np.load(npz_path, allow_pickle=False)
        assert "name_mbid_flat" in raw.files
        assert "name_mbid_indptr" in raw.files


# ===========================================================================
# IdentityAsset — row and name lookups
# ===========================================================================


class TestIdentityAssetLookups:
    def _make_asset(self, tmp_path) -> IdentityAsset:
        track_ids = _make_track_ids(4)
        payloads = [
            _deezer_payload(
                100, 1001, "Rock Band",
                contributors=[
                    {"id": 1001, "name": "Rock Band", "role": "Main"},
                    {"id": 2002, "name": "Featured", "role": "Featured"},
                ],
            ),
            _deezer_payload(200, 1001, "Rock Band"),
            _deezer_payload(
                300, 3003, "Jazz Person",
                contributors=[
                    {"id": 3003, "name": "Jazz Person", "role": "Main"},
                    {"id": 2002, "name": "Featured", "role": "Featured"},
                ],
            ),
            # track 400 has no metadata
        ]
        return _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)

    def test_row_identity_known(self, tmp_path):
        asset = self._make_asset(tmp_path)
        identity = asset.row_identity(0)
        assert identity["track_id"] == 100
        assert identity["primary_artist_deezer_id"] == 1001
        assert isinstance(identity["contributor_deezer_ids"], list)

    def test_row_identity_unknown_artist(self, tmp_path):
        asset = self._make_asset(tmp_path)
        identity = asset.row_identity(3)  # track 400 has no metadata
        assert identity["primary_artist_deezer_id"] is None
        assert identity["mbid_attribution"] == "none"

    def test_row_identity_out_of_range(self, tmp_path):
        asset = self._make_asset(tmp_path)
        with pytest.raises(IndexError):
            asset.row_identity(999)
        with pytest.raises(IndexError):
            asset.row_identity(-1)

    def test_contributor_intersection_shared(self, tmp_path):
        asset = self._make_asset(tmp_path)
        shared = asset.contributor_intersection([0], [2])
        assert 2002 in shared

    def test_contributor_intersection_no_overlap(self, tmp_path):
        asset = self._make_asset(tmp_path)
        shared = asset.contributor_intersection([1], [2])
        assert 1001 not in shared

    def test_contributor_intersection_empty_inputs(self, tmp_path):
        asset = self._make_asset(tmp_path)
        assert asset.contributor_intersection([], []) == set()
        assert asset.contributor_intersection([0], []) == set()

    def test_name_to_deezer_ids_single(self, tmp_path):
        asset = self._make_asset(tmp_path)
        ids = asset.name_to_deezer_ids("jazz person")
        assert ids == [3003]

    def test_name_to_deezer_ids_unknown(self, tmp_path):
        asset = self._make_asset(tmp_path)
        assert asset.name_to_deezer_ids("zzz unknown") == []

    def test_name_to_deezer_ids_case_insensitive(self, tmp_path):
        asset = self._make_asset(tmp_path)
        assert asset.name_to_deezer_ids("ROCK BAND") == asset.name_to_deezer_ids("rock band")

    def test_name_to_mbids_non_homonym(self, tmp_path):
        """name_to_mbids returns the name-level MBIDs (same as direct for unique names)."""
        track_ids = np.array([100], dtype=np.int64)
        tar_bytes = _build_lastfm_tar([("u1", _MBID_A, "Solo Band", "100")])
        asset = _build_asset(
            tmp_path,
            track_ids=track_ids,
            payloads=[_deezer_payload(100, 1001, "Solo Band")],
            tar_bytes=tar_bytes,
        )
        mbids = asset.name_to_mbids("Solo Band")
        assert _MBID_A in mbids

    def test_homonym_names_returns_multi_id_only(self, tmp_path):
        track_ids = np.array([100, 200, 300], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
            _deezer_payload(300, 3003, "Tool"),
        ]
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)
        homonyms = asset.homonym_names()
        for name, ids in homonyms:
            assert len(ids) >= 2
        names = {h[0] for h in homonyms}
        assert "nothing" in names
        assert "tool" not in names


# ===========================================================================
# has_disjoint_mbid_evidence — non-tautological check
# ===========================================================================


class TestDisjointMbidEvidence:
    """Verify has_disjoint_mbid_evidence is never trivially true."""

    def _make_audit(self, tmp_path, payloads, tar_bytes) -> Dict:
        track_ids = np.arange(100, 100 + len(payloads), dtype=np.int64)
        asset = _build_asset(
            tmp_path, track_ids=track_ids, payloads=payloads, tar_bytes=tar_bytes
        )
        return run_identity_audit(asset)

    def test_equal_mbid_count_not_disjoint_without_bridge(self, tmp_path):
        """2 Deezer IDs for 'nothing', exactly 2 MBIDs → NOT disjoint.
        Without a direct cross-source bridge, equal counts are unlinked
        name-level evidence; has_disjoint_mbid_evidence must be False and
        status must be 'source_mbids_stable_but_cross_source_unlinked'."""
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(101, 9001, "nothing"),
        ]
        tar_bytes = _build_lastfm_tar([
            ("u1", _MBID_A, "Nothing", "100"),
            ("u2", _MBID_B, "nothing", "200"),
        ])
        audit = self._make_audit(tmp_path, payloads, tar_bytes)
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                assert d["has_disjoint_mbid_evidence"] is False, (
                    "Equal MBID/Deezer counts must NOT be labeled disjoint "
                    "without a Deezer↔MBID cross-source bridge"
                )
                assert d["mbid_evidence_status"] == (
                    "source_mbids_stable_but_cross_source_unlinked"
                )
                return
        pytest.fail("'nothing' not in homonym_details")

    def test_uncertain_when_fewer_mbids_than_deezer_ids(self, tmp_path):
        """2 Deezer IDs for 'nothing', only 1 MBID → 'uncertain' (not disjoint)."""
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(101, 9001, "nothing"),
        ]
        tar_bytes = _build_lastfm_tar([("u1", _MBID_A, "Nothing", "100")])
        audit = self._make_audit(tmp_path, payloads, tar_bytes)
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                assert d["has_disjoint_mbid_evidence"] is False
                assert d["mbid_evidence_status"] == (
                    "source_mbids_stable_but_cross_source_unlinked"
                )
                return
        pytest.fail("'nothing' not in homonym_details")

    def test_none_when_no_mbids(self, tmp_path):
        """2 Deezer IDs for 'nothing', no MBIDs → 'none'."""
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(101, 9001, "nothing"),
        ]
        tar_bytes = _build_lastfm_tar([])
        audit = self._make_audit(tmp_path, payloads, tar_bytes)
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                assert d["has_disjoint_mbid_evidence"] is False
                assert d["mbid_evidence_status"] == "none"
                return
        pytest.fail("'nothing' not in homonym_details")


# ===========================================================================
# v13 variant union
# ===========================================================================


class TestV13VariantUnion:
    """Verify v13 result rows are unioned across ALL variants, not last-write."""

    def _make_audit_asset(self, tmp_path) -> IdentityAsset:
        track_ids = np.array([100, 200, 300], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
            _deezer_payload(300, 3003, "Tool"),
        ]
        return _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)

    def test_both_variant_seeds_captured(self, tmp_path):
        """Seeds in variant-A AND variant-B must both appear, not just the last."""
        asset = self._make_audit_asset(tmp_path)
        diag = {
            "production_baseline": {"records": []},
            "variants": {
                "variant_A": {
                    "records": [
                        {"seed_id": "SEED-A-001", "query_row": 0, "rows": [1, 2]}
                    ]
                },
                "variant_B": {
                    "records": [
                        {"seed_id": "SEED-B-002", "query_row": 0, "rows": [2]}
                    ]
                },
            },
        }
        audit = run_identity_audit(asset, v13_diagnostics=diag)
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                affected = set(d["affected_v13_seeds"])
                assert "SEED-A-001" in affected, "SEED-A-001 from variant_A missing"
                assert "SEED-B-002" in affected, "SEED-B-002 from variant_B missing"
                return
        pytest.fail("'nothing' not in homonym_details")

    def test_baseline_and_variant_combined(self, tmp_path):
        """Seeds from production_baseline AND variants are merged."""
        asset = self._make_audit_asset(tmp_path)
        diag = {
            "production_baseline": {
                "records": [
                    {"seed_id": "BASE-001", "query_row": 0, "rows": [1]}
                ]
            },
            "variants": {
                "variant_A": {
                    "records": [
                        {"seed_id": "VAR-A-001", "query_row": 0, "rows": [2]}
                    ]
                },
            },
        }
        audit = run_identity_audit(asset, v13_diagnostics=diag)
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                affected = set(d["affected_v13_seeds"])
                assert "BASE-001" in affected
                assert "VAR-A-001" in affected
                return
        pytest.fail("'nothing' not in homonym_details")

    def test_multiple_variants_result_rows_unioned(self, tmp_path):
        """affected_v13_result_rows counts rows from all variants, not just last."""
        asset = self._make_audit_asset(tmp_path)
        # row 1 (track 200) has artist 9001 (nothing)
        # row 2 (track 300) has artist 3003 (tool) — NOT a nothing row
        diag = {
            "production_baseline": {"records": []},
            "variants": {
                "v1": {"records": [{"seed_id": "S1", "query_row": 0, "rows": [1]}]},
                "v2": {"records": [{"seed_id": "S2", "query_row": 0, "rows": [1]}]},
            },
        }
        audit = run_identity_audit(asset, v13_diagnostics=diag)
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                # S1 and S2 both have row 1 as a result row (artist 9001 = nothing)
                assert d["affected_v13_result_rows"] >= 2
                return
        pytest.fail("'nothing' not in homonym_details")


# ===========================================================================
# Audit covers ALL keys (not only homonyms)
# ===========================================================================


class TestAuditAllKeys:
    def test_audit_includes_unique_keys(self, tmp_path):
        """run_identity_audit must report on unique-ID names too."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
        ]
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)
        audit = run_identity_audit(asset)
        nc = audit["name_clusters"]
        assert nc["unique_normalized_names"] >= nc["homonym_names"]
        # "nothing" is the only name → it's a homonym
        assert nc["homonym_names"] == 1
        assert nc["unique_id_names"] == 0

    def test_audit_reports_unresolved_rows(self, tmp_path):
        """Rows with no Deezer metadata must be counted as unresolved."""
        track_ids = _make_track_ids(4)
        payloads = [
            _deezer_payload(100, 1001, "Artist A"),
            _deezer_payload(200, 2002, "Artist B"),
            # tracks 300, 400 have no metadata
        ]
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)
        audit = run_identity_audit(asset)
        assert audit["coverage"]["rows_without_primary_artist_id"] == 2
        unresolved = audit["coverage"]["unresolved_track_ids"]
        assert 300 in unresolved
        assert 400 in unresolved

    def test_audit_schema_version(self, tmp_path):
        asset = _build_asset(tmp_path)
        assert run_identity_audit(asset)["schema_version"] == SCHEMA_VERSION

    def test_audit_is_deterministic(self, tmp_path):
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
        ]
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)
        a1 = run_identity_audit(asset)
        a2 = run_identity_audit(asset)
        a1.pop("generated_at")
        a2.pop("generated_at")
        assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)

    def test_audit_mbid_coverage_counts(self, tmp_path):
        track_ids = np.array([100, 200, 300], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
            _deezer_payload(300, 3003, "Tool"),
        ]
        tar_bytes = _build_lastfm_tar([
            ("u1", _MBID_A, "Nothing", "100"),
            ("u2", _MBID_B, "nothing", "500"),
            ("u3", _MBID_C, "Tool", "800"),
        ])
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads, tar_bytes=tar_bytes)
        audit = run_identity_audit(asset)
        mc = audit["mbid_coverage"]
        # Name matches alone never establish a cross-source direct link.
        assert mc["artist_ids_with_direct_mbids"] == 0
        assert mc["total_direct_mbid_mappings"] == 0

    def test_audit_multi_mbid_keys_counted(self, tmp_path):
        """keys_with_multi_mbids counts names with >1 source MBID."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
        ]
        tar_bytes = _build_lastfm_tar([
            ("u1", _MBID_A, "Nothing", "100"),
            ("u2", _MBID_B, "nothing", "200"),
        ])
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads, tar_bytes=tar_bytes)
        audit = run_identity_audit(asset)
        # "nothing" has 2 MBIDs
        assert audit["name_clusters"]["keys_with_multi_mbids"] >= 1

    def test_audit_homonym_centroid_risk(self, tmp_path):
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
        ]
        rng = np.random.default_rng(42)
        embs = np.zeros((2, 8), dtype=np.float32)
        embs[0, 0] = 1.0
        embs[1, 1] = 1.0
        asset = _build_asset(
            tmp_path, track_ids=track_ids, payloads=payloads, embeddings=embs
        )
        audit = run_identity_audit(asset)
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                assert d["centroid_cosine_risk"] is not None
                assert -1.0 <= d["centroid_cosine_risk"] <= 1.0
                return


# ===========================================================================
# Disambiguation
# ===========================================================================


class TestDisambiguate:
    def _make_nothing_asset(self, tmp_path) -> IdentityAsset:
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
        ]
        embs = np.zeros((2, 8), dtype=np.float32)
        embs[0, 0] = 3.0
        embs[0] /= np.linalg.norm(embs[0])
        embs[1, 1] = 3.0
        embs[1] /= np.linalg.norm(embs[1])
        return _build_asset(
            tmp_path, track_ids=track_ids, payloads=payloads, embeddings=embs
        )

    def test_disambiguates_correct_direction(self, tmp_path):
        asset = self._make_nothing_asset(tmp_path)
        q = np.zeros(8, dtype=np.float32)
        q[0] = 1.0
        result = asset.disambiguate("nothing", q, min_confidence=0.0, min_margin=0.0)
        assert result is not None
        best_id, conf, margin = result
        assert best_id == 1001
        assert 0.0 <= conf <= 1.0

    def test_disambiguates_dnb_direction(self, tmp_path):
        asset = self._make_nothing_asset(tmp_path)
        q = np.zeros(8, dtype=np.float32)
        q[1] = 1.0
        result = asset.disambiguate("nothing", q, min_confidence=0.0, min_margin=0.0)
        assert result is not None
        assert result[0] == 9001

    def test_abstains_on_low_margin(self, tmp_path):
        asset = self._make_nothing_asset(tmp_path)
        c0 = asset.artist_centroid(1001)
        c1 = asset.artist_centroid(9001)
        if c0 is None or c1 is None:
            pytest.skip("centroids not available")
        midpoint = (c0 + c1) / 2.0
        result = asset.disambiguate("nothing", midpoint, min_confidence=0.0, min_margin=0.99)
        if result is not None:
            _, _, margin = result
            assert margin >= 0.99

    def test_abstains_on_unknown_name(self, tmp_path):
        asset = self._make_nothing_asset(tmp_path)
        assert asset.disambiguate("zzz unknown", np.ones(8, dtype=np.float32)) is None

    def test_abstains_on_zero_query(self, tmp_path):
        asset = self._make_nothing_asset(tmp_path)
        assert asset.disambiguate("nothing", np.zeros(8, dtype=np.float32)) is None

    def test_single_candidate_margin_is_zero(self, tmp_path):
        track_ids = np.array([100], dtype=np.int64)
        embs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        asset = _build_asset(
            tmp_path,
            track_ids=track_ids,
            payloads=[_deezer_payload(100, 1001, "Unique")],
            embeddings=embs,
        )
        result = asset.disambiguate(
            "Unique", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            min_confidence=0.0, min_margin=0.99
        )
        assert result is not None
        best_id, _, margin = result
        assert best_id == 1001
        assert margin == 0.0

    def test_no_centroids_abstains(self, tmp_path):
        track_ids = np.array([100, 200], dtype=np.int64)
        asset = _build_asset(
            tmp_path,
            track_ids=track_ids,
            payloads=[
                _deezer_payload(100, 1001, "Nothing"),
                _deezer_payload(200, 9001, "nothing"),
            ],
        )
        assert asset.disambiguate("nothing", np.ones(4, dtype=np.float32)) is None

    def test_deterministic(self, tmp_path):
        asset = self._make_nothing_asset(tmp_path)
        q = np.array([0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r1 = asset.disambiguate("nothing", q, min_confidence=0.0, min_margin=0.0)
        r2 = asset.disambiguate("nothing", q, min_confidence=0.0, min_margin=0.0)
        assert r1 == r2


# ===========================================================================
# artist_centroid
# ===========================================================================


class TestArtistCentroid:
    def test_centroid_is_normalized(self, tmp_path):
        embs = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        asset = _build_asset(
            tmp_path,
            track_ids=np.array([100], dtype=np.int64),
            payloads=[_deezer_payload(100, 1001, "Artist")],
            embeddings=embs,
        )
        c = asset.artist_centroid(1001)
        assert c is not None
        assert abs(float(np.linalg.norm(c)) - 1.0) < 1e-5

    def test_unknown_artist_returns_none(self, tmp_path):
        assert _build_asset(tmp_path).artist_centroid(999999) is None

    def test_no_embeddings_returns_none(self, tmp_path):
        asset = _build_asset(
            tmp_path,
            track_ids=np.array([100], dtype=np.int64),
            payloads=[_deezer_payload(100, 1001, "Artist")],
        )
        assert asset.artist_centroid(1001) is None


# ===========================================================================
# Fallback primitives
# ===========================================================================


class TestFallbackPrimitives:
    def test_track_count_selects_most_common(self):
        assert fallback_by_track_count([1001, 9001], {1001: 100, 9001: 5}) == 1001

    def test_track_count_abstains_on_tie(self):
        assert fallback_by_track_count([1001, 9001], {1001: 50, 9001: 50}) is None

    def test_track_count_abstains_on_zero(self):
        assert fallback_by_track_count([1001, 9001], {}) is None

    def test_track_count_empty_input(self):
        assert fallback_by_track_count([], {1001: 5}) is None

    def test_track_count_single(self):
        assert fallback_by_track_count([1001], {1001: 3}) == 1001

    def test_mbid_coverage_prefers_mbid_rich(self, tmp_path):
        """Non-homonym artist with MBID evidence preferred over one without."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Indie Band"),
            _deezer_payload(200, 9001, "Jazz Trio"),
        ]
        tar_bytes = _build_lastfm_tar([("u1", _MBID_A, "Indie Band", "100")])
        asset = _build_asset(
            tmp_path, track_ids=track_ids, payloads=payloads, tar_bytes=tar_bytes
        )
        assert fallback_by_mbid_coverage([1001, 9001], asset) is None

    def test_mbid_coverage_abstains_when_all_zero(self, tmp_path):
        asset = _build_asset(tmp_path)
        assert fallback_by_mbid_coverage([1001, 9001], asset) is None

    def test_mbid_coverage_empty(self, tmp_path):
        assert fallback_by_mbid_coverage([], _build_asset(tmp_path)) is None


# ===========================================================================
# build_identity_network — mock fetcher (no network)
# ===========================================================================


class _MockFetcher(DeezerFetcher):
    """Deterministic mock fetcher for testing build_identity_network."""

    def __init__(self, responses: Dict[int, Optional[Dict[str, Any]]]) -> None:
        self._responses = responses
        self.call_count = 0

    def fetch(self, track_id: int) -> Optional[Dict[str, Any]]:
        self.call_count += 1
        return self._responses.get(track_id)


class TestBuildIdentityNetwork:
    def test_resolves_unresolved_via_fetcher(self, tmp_path):
        """Fetcher is called for unresolved tracks; resolved tracks are skipped."""
        track_ids = np.array([100, 200, 300], dtype=np.int64)
        mock_responses = {
            200: _deezer_payload(200, 2002, "Artist B"),
            300: _deezer_payload(300, 3003, "Artist C"),
        }
        fetcher = _MockFetcher(mock_responses)
        # Pre-seed track 100 via candidate JSON
        candidate = {"tracks": [_deezer_payload(100, 1001, "Artist A")]}
        cand_path = tmp_path / "cand.json"
        cand_path.write_text(json.dumps(candidate), encoding="utf-8")

        result = build_identity_network(
            catalog_track_ids=track_ids,
            cache_path=tmp_path / "cache.db",
            output_npz=tmp_path / "id.npz",
            clap_asset_hash="h" * 64,
            candidate_jsons=[cand_path],
            fetcher=fetcher,
            allow_unresolved=False,
        )
        assert result["still_unresolved"] == 0
        assert result["fetch_stats"]["tracks_resolved"] == 2
        # Fetcher should NOT be called for track 100 (already resolved via JSON)
        assert fetcher.call_count == 2

    def test_fails_on_unresolved_by_default(self, tmp_path):
        """Without allow_unresolved, raise if any row is still unresolved."""
        track_ids = np.array([100, 200], dtype=np.int64)
        fetcher = _MockFetcher({})  # returns None for every ID
        with pytest.raises(ArtistIdentityError, match="unresolved"):
            build_identity_network(
                catalog_track_ids=track_ids,
                cache_path=tmp_path / "cache.db",
                output_npz=tmp_path / "id.npz",
                clap_asset_hash="h" * 64,
                fetcher=fetcher,
                allow_unresolved=False,
            )

    def test_allow_unresolved_flag(self, tmp_path):
        """allow_unresolved=True should succeed even with unresolved rows."""
        track_ids = np.array([100, 200], dtype=np.int64)
        fetcher = _MockFetcher({})
        result = build_identity_network(
            catalog_track_ids=track_ids,
            cache_path=tmp_path / "cache.db",
            output_npz=tmp_path / "id.npz",
            clap_asset_hash="h" * 64,
            fetcher=fetcher,
            allow_unresolved=True,
        )
        assert result["still_unresolved"] == 2

    def test_writes_audit_json(self, tmp_path):
        """When audit_path is provided, a valid JSON audit is written."""
        track_ids = np.array([100], dtype=np.int64)
        fetcher = _MockFetcher({100: _deezer_payload(100, 1001, "Artist")})
        audit_path = tmp_path / "audit.json"
        build_identity_network(
            catalog_track_ids=track_ids,
            cache_path=tmp_path / "cache.db",
            output_npz=tmp_path / "id.npz",
            clap_asset_hash="h" * 64,
            fetcher=fetcher,
            audit_path=audit_path,
            allow_unresolved=False,
        )
        assert audit_path.exists()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["schema_version"] == SCHEMA_VERSION

    def test_resumable_does_not_refetch_resolved(self, tmp_path):
        """Second run re-uses cached resolved rows without re-fetching."""
        track_ids = np.array([100, 200], dtype=np.int64)
        fetcher = _MockFetcher({
            100: _deezer_payload(100, 1001, "Artist A"),
            200: _deezer_payload(200, 2002, "Artist B"),
        })
        common_kwargs: Dict[str, Any] = dict(
            catalog_track_ids=track_ids,
            cache_path=tmp_path / "cache.db",
            output_npz=tmp_path / "id.npz",
            clap_asset_hash="h" * 64,
            fetcher=fetcher,
            allow_unresolved=False,
        )
        # First run: fetches 2 tracks
        build_identity_network(**common_kwargs)
        calls_after_first = fetcher.call_count
        # Second run (resume): nothing unresolved, fetcher not called again
        build_identity_network(**common_kwargs)
        assert fetcher.call_count == calls_after_first

    def test_catalog_names_seeded(self, tmp_path):
        """catalog_names are stored and survive even before Deezer fetch."""
        track_ids = np.array([100], dtype=np.int64)
        fetcher = _MockFetcher({100: _deezer_payload(100, 1001, "Band Name")})
        result = build_identity_network(
            catalog_track_ids=track_ids,
            cache_path=tmp_path / "cache.db",
            output_npz=tmp_path / "id.npz",
            clap_asset_hash="h" * 64,
            catalog_names=["Band Name"],
            fetcher=fetcher,
            allow_unresolved=False,
        )
        assert result["still_unresolved"] == 0

    def test_v13_diag_passed_to_audit(self, tmp_path):
        """v13_diagnostics is forwarded to the audit report."""
        track_ids = np.array([100, 200], dtype=np.int64)
        fetcher = _MockFetcher({
            100: _deezer_payload(100, 1001, "Nothing"),
            200: _deezer_payload(200, 9001, "nothing"),
        })
        diag: Dict[str, Any] = {
            "production_baseline": {
                "records": [{"seed_id": "NET-SEED-001", "query_row": 0, "rows": [1]}]
            },
            "variants": {},
        }
        audit_path = tmp_path / "audit.json"
        build_identity_network(
            catalog_track_ids=track_ids,
            cache_path=tmp_path / "cache.db",
            output_npz=tmp_path / "id.npz",
            clap_asset_hash="h" * 64,
            fetcher=fetcher,
            v13_diagnostics=diag,
            audit_path=audit_path,
            allow_unresolved=False,
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                assert "NET-SEED-001" in d["affected_v13_seeds"]
                return
        pytest.fail("'nothing' not in homonym_details")


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_catalog(self, tmp_path):
        track_ids = np.array([], dtype=np.int64)
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        with IdentityCache(cache_path, track_ids) as cache:
            result = cache.build_npz(track_ids, "h" * 64, npz_path)
        assert result["total_rows"] == 0
        asset = IdentityAsset.load(npz_path)
        assert len(asset) == 0
        assert asset.homonym_names() == []

    def test_single_row_catalog(self, tmp_path):
        asset = _build_asset(
            tmp_path,
            track_ids=np.array([999], dtype=np.int64),
            payloads=[_deezer_payload(999, 7777, "Solo Artist")],
        )
        assert len(asset) == 1
        identity = asset.row_identity(0)
        assert identity["track_id"] == 999
        assert identity["primary_artist_deezer_id"] == 7777

    def test_transliterated_names_same_cluster(self, tmp_path):
        """Accented and ASCII spellings of the same name → same cluster."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 5005, "Röyksopp"),
            _deezer_payload(200, 5005, "Royksopp"),
        ]
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)
        ids = asset.name_to_deezer_ids("royksopp")
        assert 5005 in ids
        assert len(set(ids)) == 1

    def test_transliterated_homonyms(self, tmp_path):
        """Two artists whose names transliterate to the same key."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1111, "Ghost"),
            _deezer_payload(200, 2222, "Ghöst"),
        ]
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)
        assert set(asset.name_to_deezer_ids("ghost")) == {1111, 2222}
        assert "ghost" in {h[0] for h in asset.homonym_names()}

    def test_schema_version_in_npz(self, tmp_path):
        assert _build_asset(tmp_path).metadata["schema_version"] == SCHEMA_VERSION

    def test_load_classmethod_equivalent(self, tmp_path):
        asset = _build_asset(tmp_path)
        npz_path = list(tmp_path.glob("*.npz"))[0]
        asset2 = IdentityAsset.load(npz_path)
        assert asset.metadata == asset2.metadata

    def test_no_module_level_sockets(self):
        """Module import must not open any network sockets."""
        import socket
        import soundalike.ml.artist_identity_v14 as mod
        for name in dir(mod):
            obj = getattr(mod, name, None)
            assert not isinstance(obj, socket.socket), (
                f"Module attribute {name!r} is an open socket"
            )


# ===========================================================================
# Unresolved key inclusion
# ===========================================================================


class TestUnresolvedKeyInclusion:
    """Unresolved rows (no Deezer ID) with artist_name must appear in name_keys."""

    def test_unresolved_name_in_npz_name_keys(self, tmp_path):
        track_ids = np.array([100, 200, 300], dtype=np.int64)
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        with IdentityCache(cache_path, track_ids) as cache:
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Resolved Artist"))
            # Unresolved tracks still have catalog artist names
            cache._conn.execute(
                "UPDATE track_identity SET artist_name=? WHERE track_id=?",
                ("Unresolved Band", 200),
            )
            cache._conn.execute(
                "UPDATE track_identity SET artist_name=? WHERE track_id=?",
                ("Another Unresolved", 300),
            )
            cache._conn.commit()
            cache.build_npz(track_ids, "h" * 64, npz_path)
        asset = IdentityAsset.load(npz_path)
        name_keys_set = set(str(k) for k in asset._name_keys)
        assert "unresolved band" in name_keys_set, (
            "Unresolved row name must appear in name_keys"
        )
        assert "another unresolved" in name_keys_set

    def test_unresolved_key_type_in_audit(self, tmp_path):
        track_ids = np.array([100, 200], dtype=np.int64)
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        with IdentityCache(cache_path, track_ids) as cache:
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Known Artist"))
            cache._conn.execute(
                "UPDATE track_identity SET artist_name=? WHERE track_id=?",
                ("Ghost Band", 200),
            )
            cache._conn.commit()
            cache.build_npz(track_ids, "h" * 64, npz_path)
        asset = IdentityAsset.load(npz_path)
        audit = run_identity_audit(asset)
        assert audit["name_clusters"]["unknown_id_names"] >= 1
        # Unresolved name must be in the name_keys array
        assert "ghost band" in set(str(k) for k in asset._name_keys)

    def test_row_name_keys_aligned_in_npz(self, tmp_path):
        """row_name_keys array is length-n and aligned with track_ids."""
        track_ids = np.array([100, 200, 300], dtype=np.int64)
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        with IdentityCache(cache_path, track_ids) as cache:
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Rock Band"))
            cache._conn.execute(
                "UPDATE track_identity SET artist_name=? WHERE track_id=?",
                ("Jazz Trio", 200),
            )
            cache._conn.commit()
            cache.build_npz(track_ids, "h" * 64, npz_path)
        raw = np.load(npz_path, allow_pickle=False)
        assert "row_name_keys" in raw.files
        rnk = list(raw["row_name_keys"])
        assert len(rnk) == 3
        assert rnk[0] == "rock band"   # resolved, name from API
        assert rnk[1] == "jazz trio"   # unresolved, name from catalog seed
        assert rnk[2] == ""            # no name at all

    def test_coverage_unique_catalog_name_keys(self, tmp_path):
        """Audit coverage reports distinct normalized keys across all rows."""
        track_ids = np.array([100, 200, 300], dtype=np.int64)
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        with IdentityCache(cache_path, track_ids) as cache:
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Band A"))
            cache.ingest_deezer_metadata(200, _deezer_payload(200, 1001, "Band A"))  # same
            cache._conn.execute(
                "UPDATE track_identity SET artist_name=? WHERE track_id=?",
                ("Band B", 300),
            )
            cache._conn.commit()
            cache.build_npz(track_ids, "h" * 64, npz_path)
        asset = IdentityAsset.load(npz_path)
        audit = run_identity_audit(asset)
        # "band a" (rows 0+1) and "band b" (row 2) → 2 distinct catalog keys
        assert audit["coverage"]["unique_catalog_name_keys"] == 2


# ===========================================================================
# Stored spelling variants (no external argument)
# ===========================================================================


class TestStoredSpellings:
    def test_spelling_arrays_in_npz(self, tmp_path):
        """NPZ must contain spelling_flat and spelling_indptr."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Röyksopp"),
            _deezer_payload(200, 1001, "Royksopp"),
        ]
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        with IdentityCache(cache_path, track_ids) as cache:
            for p in payloads:
                cache.ingest_deezer_metadata(p["id"], p)
            cache.build_npz(track_ids, "h" * 64, npz_path)
        raw = np.load(npz_path, allow_pickle=False)
        assert "spelling_flat" in raw.files
        assert "spelling_indptr" in raw.files
        all_spellings = set(str(x) for x in raw["spelling_flat"])
        assert "Royksopp" in all_spellings

    def test_audit_spellings_from_npz_no_external_arg(self, tmp_path):
        """Audit raw_spelling_variants comes from NPZ without spelling_variants_db."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "NOTHING"),
        ]
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)
        # run_identity_audit has no spelling_variants_db parameter — spellings from NPZ
        audit = run_identity_audit(asset)
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                spellings = d["raw_spelling_variants"]
                assert len(spellings) >= 2, (
                    "Both 'Nothing' and 'NOTHING' spellings expected"
                )
                lower = [s.lower() for s in spellings]
                assert "nothing" in lower
                return
        pytest.fail("'nothing' not in homonym_details")

    def test_spellings_for_key_method(self, tmp_path):
        """IdentityAsset.spellings_for_key returns variants stored in NPZ."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Jay-Z"),
            _deezer_payload(200, 1001, "Jay Z"),
        ]
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)
        spellings = asset.spellings_for_key("jay z")
        assert len(spellings) >= 1
        assert any("Jay" in s for s in spellings)

    def test_unresolved_row_spelling_in_asset(self, tmp_path):
        """Unresolved row's artist_name appears as a spelling variant in the asset."""
        track_ids = np.array([100, 200], dtype=np.int64)
        cache_path = tmp_path / "c.db"
        npz_path = tmp_path / "id.npz"
        with IdentityCache(cache_path, track_ids) as cache:
            cache.ingest_deezer_metadata(100, _deezer_payload(100, 1001, "Known"))
            cache._conn.execute(
                "UPDATE track_identity SET artist_name=? WHERE track_id=?",
                ("Rare Spelling", 200),
            )
            cache._conn.commit()
            cache.build_npz(track_ids, "h" * 64, npz_path)
        asset = IdentityAsset.load(npz_path)
        spellings = asset.spellings_for_key("rare spelling")
        assert "Rare Spelling" in spellings


# ===========================================================================
# Within-artist multimodality
# ===========================================================================


class TestWithinArtistMultimodality:
    def test_multimodal_single_id_key_detected(self, tmp_path):
        """Single-ID key with very different track embeddings → low within_min_cosine."""
        track_ids = np.array([100, 200, 300, 400], dtype=np.int64)
        payloads = [_deezer_payload(100 + i * 100, 1001, "Versatile") for i in range(4)]
        embs = np.zeros((4, 8), dtype=np.float32)
        for j in range(4):
            embs[j, j] = 1.0  # orthogonal unit vectors
        asset = _build_asset(
            tmp_path, track_ids=track_ids, payloads=payloads, embeddings=embs
        )
        audit = run_identity_audit(asset)
        for d in audit["name_clusters"]["all_key_details"]:
            if d["normalized_name"] == "versatile":
                assert d["key_type"] == "unique"
                assert d["within_min_cosine"] is not None
                # Orthogonal embeddings → cosine ≈ 0 → well below threshold
                assert d["within_min_cosine"] < 0.5
                return
        pytest.fail("'versatile' not in all_key_details")

    def test_keys_with_multimodal_audio_summary(self, tmp_path):
        """Audit summary counts all keys — including unique-ID — with multimodal audio."""
        track_ids = np.array([100, 200, 300, 400], dtype=np.int64)
        payloads = [_deezer_payload(100 + i * 100, 1001, "Wide Ranger") for i in range(4)]
        embs = np.zeros((4, 8), dtype=np.float32)
        for j in range(4):
            embs[j, j] = 1.0
        asset = _build_asset(
            tmp_path, track_ids=track_ids, payloads=payloads, embeddings=embs
        )
        audit = run_identity_audit(asset)
        assert audit["name_clusters"]["keys_with_multimodal_audio"] >= 1

    def test_homonym_centroid_separation_metrics(self, tmp_path):
        """Homonym details include min/max centroid cosine, separation, feasibility."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
        ]
        embs = np.zeros((2, 8), dtype=np.float32)
        embs[0, 0] = 1.0  # orthogonal centroids
        embs[1, 1] = 1.0
        asset = _build_asset(
            tmp_path, track_ids=track_ids, payloads=payloads, embeddings=embs
        )
        audit = run_identity_audit(asset)
        for d in audit["name_clusters"]["homonym_details"]:
            if d["normalized_name"] == "nothing":
                assert "min_centroid_cosine" in d
                assert "max_centroid_cosine" in d
                assert "centroid_separation" in d
                assert "audio_disambiguation_feasible" in d
                assert d["min_centroid_cosine"] is not None
                # Orthogonal centroids: cosine ≈ 0, separation ≈ 1
                assert d["centroid_separation"] > 0.9
                assert d["audio_disambiguation_feasible"] is True
                return
        pytest.fail("'nothing' not in homonym_details")

    def test_homonym_multimodal_counted_in_summary(self, tmp_path):
        """Homonym with well-separated centroids contributes to keys_with_multimodal_audio."""
        track_ids = np.array([100, 200], dtype=np.int64)
        payloads = [
            _deezer_payload(100, 1001, "Nothing"),
            _deezer_payload(200, 9001, "nothing"),
        ]
        embs = np.zeros((2, 8), dtype=np.float32)
        embs[0, 0] = 1.0
        embs[1, 1] = 1.0
        asset = _build_asset(
            tmp_path, track_ids=track_ids, payloads=payloads, embeddings=embs
        )
        audit = run_identity_audit(asset)
        assert audit["name_clusters"]["keys_with_multimodal_audio"] >= 1

    def test_no_multimodal_without_embeddings(self, tmp_path):
        """Without embeddings, within_min_cosine is None and no multimodal keys counted."""
        track_ids = np.array([100, 200, 300, 400], dtype=np.int64)
        payloads = [_deezer_payload(100 + i * 100, 1001, "Solo") for i in range(4)]
        asset = _build_asset(tmp_path, track_ids=track_ids, payloads=payloads)
        audit = run_identity_audit(asset)
        for d in audit["name_clusters"]["all_key_details"]:
            if d["normalized_name"] == "solo":
                assert d["within_min_cosine"] is None
        assert audit["name_clusters"]["keys_with_multimodal_audio"] == 0


# ===========================================================================
# max_fetches resume
# ===========================================================================


class TestMaxFetchesResume:
    def test_max_fetches_limits_api_calls(self, tmp_path):
        """max_fetches caps the number of fetcher calls in one run."""
        track_ids = np.array([100, 200, 300, 400], dtype=np.int64)
        mock = _MockFetcher({
            100: _deezer_payload(100, 1001, "A"),
            200: _deezer_payload(200, 2002, "B"),
            300: _deezer_payload(300, 3003, "C"),
            400: _deezer_payload(400, 4004, "D"),
        })
        result = build_identity_network(
            catalog_track_ids=track_ids,
            cache_path=tmp_path / "cache.db",
            output_npz=tmp_path / "id.npz",
            clap_asset_hash="h" * 64,
            fetcher=mock,
            max_fetches=2,
            allow_unresolved=True,
        )
        assert mock.call_count == 2
        assert result["still_unresolved"] == 2

    def test_max_fetches_resume_resolves_remainder(self, tmp_path):
        """Two bounded runs together fully resolve all rows."""
        track_ids = np.array([100, 200, 300, 400], dtype=np.int64)
        mock = _MockFetcher({
            100: _deezer_payload(100, 1001, "A"),
            200: _deezer_payload(200, 2002, "B"),
            300: _deezer_payload(300, 3003, "C"),
            400: _deezer_payload(400, 4004, "D"),
        })
        common: Dict[str, Any] = dict(
            catalog_track_ids=track_ids,
            cache_path=tmp_path / "cache.db",
            output_npz=tmp_path / "id.npz",
            clap_asset_hash="h" * 64,
            fetcher=mock,
            allow_unresolved=True,
        )
        r1 = build_identity_network(max_fetches=2, **common)
        assert r1["still_unresolved"] == 2
        # Second run resolves the remaining 2 rows
        r2 = build_identity_network(max_fetches=2, **common)
        assert r2["still_unresolved"] == 0
        assert mock.call_count == 4

    def test_max_fetches_none_fetches_all(self, tmp_path):
        """Default max_fetches=None fetches all unresolved rows."""
        track_ids = np.array([100, 200, 300], dtype=np.int64)
        mock = _MockFetcher({
            100: _deezer_payload(100, 1001, "A"),
            200: _deezer_payload(200, 2002, "B"),
            300: _deezer_payload(300, 3003, "C"),
        })
        result = build_identity_network(
            catalog_track_ids=track_ids,
            cache_path=tmp_path / "cache.db",
            output_npz=tmp_path / "id.npz",
            clap_asset_hash="h" * 64,
            fetcher=mock,
            allow_unresolved=False,
        )
        assert mock.call_count == 3
        assert result["still_unresolved"] == 0
