from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from soundalike.ml.fulltrack_pilot import (
    METHODS,
    MODEL_FAMILIES,
    PILOT_SCENES,
    FullTrackPilotError,
    PilotConfig,
    SeedCandidate,
    build_blinded_documents,
    diversity_evidence,
    lawful_stream_url,
    select_diverse_seeds,
    validate_blinded_documents,
)
from soundalike.ml.jamendo_fulltrack import JamendoTrack, TrackLicense
from soundalike.ml.fulltrack_store import stable_json_sha256


HASH = "a" * 64


def _candidate(index: int) -> SeedCandidate:
    tempo = (72.0, 110.0, 148.0)[index % 3]
    return SeedCandidate(
        track_id=1000 + index,
        artist_id=2000 + index,
        tags=(PILOT_SCENES[index], f"mood---mood-{index}"),
        tempo_bpm=tempo,
        texture_region=index % 5,
    )


def _track(track_id: int, artist_id: int) -> JamendoTrack:
    path = f"{track_id // 1000:02d}/{track_id}.mp3"
    return JamendoTrack(
        row_index=track_id,
        track_id=track_id,
        artist_id=artist_id,
        album_id=artist_id + 10_000,
        relative_path=path,
        audio_path=Path("X:/lawful-source") / path,
        duration_seconds=180.0,
        tags=(),
        title=f"Track {track_id}",
        artist_name=f"Artist {artist_id}",
        album_name=f"Album {artist_id}",
        release_date="2020-01-01",
        jamendo_url=f"https://www.jamendo.com/track/{track_id}",
        license=TrackLicense(
            path=path,
            attribution=f"Track {track_id} by Artist {artist_id}, CC BY 3.0",
            name="CC BY 3.0",
            url="https://creativecommons.org/licenses/by/3.0/",
            permits_commercial_use=True,
            permits_derivatives=True,
        ),
        expected_audio_sha256=HASH,
        expected_audio_bytes=1_000_000,
    )


def _binding(
    family: str, store_binding_sha256: str, *, model_seed: int = 17
) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_kind=family,
        seed=model_seed,
        fold_index=0,
        report_sha256=HASH,
        model_artifact_sha256="b" * 64,
        model_json_sha256="c" * 64,
        weights_npz_sha256="d" * 64,
        source_fingerprint="e" * 64,
        store_binding_sha256=store_binding_sha256,
        training_config_sha256="1" * 64,
        job_config_sha256="2" * 64,
        maxsim_budget=8,
        embedding_dim=512,
    )


def _documents(*, fold_part: str = "test", model_seed: int = 17):
    seeds = tuple(_candidate(index) for index in range(20))
    tracks = {seed.track_id: _track(seed.track_id, seed.artist_id) for seed in seeds}
    rankings = {}
    next_track = 10_000
    for seed in seeds:
        ranked = []
        for _ in range(5):
            tracks[next_track] = _track(next_track, next_track + 50_000)
            ranked.append(next_track)
            next_track += 1
        rankings[seed.track_id] = {method: tuple(ranked) for method in METHODS}
    store_rows = {track_id: row for row, track_id in enumerate(sorted(tracks))}
    verification = {
        lawful_stream_url(track_id): {
            "status": 200,
            "content_type": "audio/mpeg",
            "content_length": 1_000_000,
            "accept_ranges": "bytes",
            "final_host": "prod-1.storage.jamendo.com",
        }
        for track_id in tracks
    }
    store_binding = {
        "schema_version": 2,
        "source_fingerprint": "e" * 64,
        "config_sha256": "3" * 64,
        "model_sha256": "4" * 64,
        "model_id": "laion-clap-630k-audioset-best",
        "embedding_dim": 512,
        "track_count": 55_701,
        "shard_tracks": 256,
        "repetition_sections": 32,
        "salient_sections": 32,
        "track_plan_sha256": "5" * 64,
        "sealed_manifest_sha256": "6" * 64,
    }
    diversity = diversity_evidence(
        seeds,
        np.eye(20, dtype=np.float32),
        texture_anchor_track_ids=[1000, 1001, 1002, 1003, 1004],
    )
    return build_blinded_documents(
        rankings=rankings,
        selected_seeds=seeds,
        tracks_by_id=tracks,
        store_rows=store_rows,
        fold_track_parts={track_id: fold_part for track_id in tracks},
        store_binding=store_binding,
        source_fingerprint="e" * 64,
        fold_index=0,
        model_bindings={
            family: _binding(
                family, stable_json_sha256(store_binding), model_seed=model_seed
            )
            for family in MODEL_FAMILIES
        },
        blinding_key=bytes(range(32)),
        diversity=diversity,
        audio_verification=verification,
    )


def test_tempo_cache_deterministically_records_invalid_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import soundalike.ml.fulltrack_pilot as pilot

    tracks = [_track(101, 201), _track(102, 202)]

    def measure(track: JamendoTrack) -> float:
        if track.track_id == 102:
            raise FullTrackPilotError("invalid descriptor")
        return 120.0

    monkeypatch.setattr(pilot, "measure_tempo_bpm", measure)
    cache = tmp_path / "tempo.private.json"
    first = pilot._load_or_measure_tempos(
        cache, source_fingerprint=HASH, shortlist=tracks
    )
    monkeypatch.setattr(
        pilot,
        "measure_tempo_bpm",
        lambda _track: (_ for _ in ()).throw(AssertionError("cache not reused")),
    )
    second = pilot._load_or_measure_tempos(
        cache, source_fingerprint=HASH, shortlist=tracks
    )

    assert first == second == {101: 120.0}
    document = __import__("json").loads(cache.read_text(encoding="utf-8"))
    assert document["rejected_track_ids"] == [102]
    assert document["track_ids"] == [101, 102]


def test_seed_selection_is_deterministic_unique_and_diverse() -> None:
    candidates = tuple(_candidate(index) for index in range(20))
    embeddings = np.eye(20, dtype=np.float32)

    first = select_diverse_seeds(candidates, embeddings)
    second = select_diverse_seeds(candidates, embeddings.copy())

    assert first == second
    assert len(first) == 20
    assert len({item.track_id for item in first}) == 20
    assert len({item.artist_id for item in first}) == 20
    evidence = diversity_evidence(
        first, embeddings, texture_anchor_track_ids=[1000, 1001, 1002, 1003, 1004]
    )
    assert evidence["scene_count"] == 20
    assert set(evidence["tempo_bpm"]["bin_counts"]) == {"slow", "medium", "fast"}
    assert set(evidence["clap_texture"]["region_counts"]) == set("01234")


def test_seed_selection_fails_closed_when_a_scene_is_missing() -> None:
    candidates = tuple(_candidate(index) for index in range(19))
    with pytest.raises(FullTrackPilotError, match="genre---world"):
        select_diverse_seeds(candidates, np.eye(19, dtype=np.float32))


def test_seed_selection_fails_closed_without_texture_coverage() -> None:
    candidates = tuple(
        SeedCandidate(
            track_id=item.track_id,
            artist_id=item.artist_id,
            tags=item.tags,
            tempo_bpm=item.tempo_bpm,
            texture_region=0,
        )
        for item in (_candidate(index) for index in range(20))
    )

    with pytest.raises(FullTrackPilotError, match="five CLAP texture regions"):
        select_diverse_seeds(candidates, np.eye(20, dtype=np.float32))


def test_blinded_pack_binds_all_methods_without_public_identity() -> None:
    public, private = _documents()

    validate_blinded_documents(public, private)
    assert public["seed_count"] == 20
    assert public["ratings_count_at_freeze"] == 0
    assert public["promotion_allowed"] is False
    public_text = str(public)
    for method in METHODS:
        assert method not in public_text
    assert private["methods"] == list(METHODS)


def test_blinded_pack_rejects_rehashed_ranking_tampering() -> None:
    public, private = _documents()
    tampered = copy.deepcopy(public)
    tampered["seeds"][0]["lists"][0]["ranking"][0]["track_id"] += 1
    tampered.pop("content_sha256")
    from soundalike.ml.fulltrack_pilot import _content_sha256

    tampered["content_sha256"] = _content_sha256(tampered)

    with pytest.raises(FullTrackPilotError, match="commitment"):
        validate_blinded_documents(tampered, private)


def test_blinded_pack_rejects_unsealed_store_binding() -> None:
    public, private = _documents()
    public = copy.deepcopy(public)
    private = copy.deepcopy(private)
    public["store_binding"].pop("sealed_manifest_sha256")
    private["store_binding"].pop("sealed_manifest_sha256")
    from soundalike.ml.fulltrack_pilot import _content_sha256

    public["store_binding_sha256"] = private["store_binding_sha256"] = (
        "0" * 64
    )
    private.pop("content_sha256")
    private["content_sha256"] = _content_sha256(private)
    public["blinding"]["private_unblinding_sha256"] = private["content_sha256"]
    public.pop("content_sha256")
    public["content_sha256"] = _content_sha256(public)

    with pytest.raises(FullTrackPilotError, match="sealed store"):
        validate_blinded_documents(public, private)


def test_blinded_pack_rejects_rehashed_license_drift() -> None:
    public, private = _documents()
    public = copy.deepcopy(public)
    first_track = next(iter(public["tracks"].values()))
    first_track["license"]["attribution"] = "drifted attribution"
    from soundalike.ml.fulltrack_pilot import _content_sha256

    public.pop("content_sha256")
    public["content_sha256"] = _content_sha256(public)

    with pytest.raises(FullTrackPilotError, match="evidence commitment"):
        validate_blinded_documents(public, private)


def test_blinded_pack_rejects_rehashed_artifact_drift() -> None:
    public, private = _documents()
    public = copy.deepcopy(public)
    private = copy.deepcopy(private)
    trained = next(
        item
        for seed in private["seeds"]
        for item in seed["lists"]
        if item["method_binding"]["trained"]
    )
    trained["method_binding"]["artifact"]["source_fingerprint"] = "9" * 64
    from soundalike.ml.fulltrack_pilot import _content_sha256

    private.pop("content_sha256")
    private["content_sha256"] = _content_sha256(private)
    public["blinding"]["private_unblinding_sha256"] = private["content_sha256"]
    public.pop("content_sha256")
    public["content_sha256"] = _content_sha256(public)

    with pytest.raises(FullTrackPilotError, match="artifact identity"):
        validate_blinded_documents(public, private)


def test_blinded_pack_rejects_rehashed_unofficial_fold() -> None:
    public, private = _documents()
    public = copy.deepcopy(public)
    private = copy.deepcopy(private)
    from soundalike.ml.fulltrack_pilot import _content_sha256

    public["fold"] = private["fold"] = 99
    private.pop("content_sha256")
    private["content_sha256"] = _content_sha256(private)
    public["blinding"]["private_unblinding_sha256"] = private["content_sha256"]
    public.pop("content_sha256")
    public["content_sha256"] = _content_sha256(public)

    with pytest.raises(FullTrackPilotError, match="strict validation"):
        validate_blinded_documents(public, private)


def test_pilot_config_rejects_artifact_budget_or_seed_drift() -> None:
    with pytest.raises(FullTrackPilotError, match="MaxSim budget"):
        PilotConfig(maxsim_budget=7).validate()
    with pytest.raises(FullTrackPilotError, match="model seed"):
        PilotConfig(model_seed=19).validate()


def test_builder_rejects_track_outside_official_test_fold() -> None:
    with pytest.raises(FullTrackPilotError, match="official held-out fold"):
        _documents(fold_part="validation")


def test_blinded_pack_rejects_rehashed_seed_schema_drift() -> None:
    public, private = _documents()
    public = copy.deepcopy(public)
    public["seeds"][0]["unexpected"] = True
    from soundalike.ml.fulltrack_pilot import _content_sha256

    public.pop("content_sha256")
    public["content_sha256"] = _content_sha256(public)

    with pytest.raises(FullTrackPilotError, match="seed schema"):
        validate_blinded_documents(public, private)


def test_blinded_pack_rejects_rehashed_seed_descriptor_drift() -> None:
    public, private = _documents()
    public = copy.deepcopy(public)
    public["seeds"][0]["tempo_bpm"] += 1.0
    from soundalike.ml.fulltrack_pilot import _content_sha256

    public.pop("content_sha256")
    public["content_sha256"] = _content_sha256(public)

    with pytest.raises(FullTrackPilotError, match="evidence commitment"):
        validate_blinded_documents(public, private)


def test_blinded_pack_rejects_rehashed_result_identity_drift() -> None:
    public, private = _documents()
    public = copy.deepcopy(public)
    public["seeds"][0]["result_ids"][0]["result_id"] = "result-" + "f" * 24
    from soundalike.ml.fulltrack_pilot import _content_sha256

    public.pop("content_sha256")
    public["content_sha256"] = _content_sha256(public)

    with pytest.raises(FullTrackPilotError, match="ranked source/fold/artist binding"):
        validate_blinded_documents(public, private)


def test_blinded_pack_rejects_rehashed_model_seed_drift() -> None:
    public, private = _documents()
    public = copy.deepcopy(public)
    private = copy.deepcopy(private)
    trained = next(
        item
        for seed in private["seeds"]
        for item in seed["lists"]
        if item["method_binding"]["trained"]
    )
    trained["method_binding"]["artifact"]["seed"] = 19
    from soundalike.ml.fulltrack_pilot import _content_sha256

    private.pop("content_sha256")
    private["content_sha256"] = _content_sha256(private)
    public["blinding"]["private_unblinding_sha256"] = private["content_sha256"]
    public.pop("content_sha256")
    public["content_sha256"] = _content_sha256(public)

    with pytest.raises(FullTrackPilotError, match="artifact identity"):
        validate_blinded_documents(public, private)
