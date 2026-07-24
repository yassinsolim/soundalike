import concurrent.futures
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import soundalike.ml.fulltrack_train as fulltrack_train
from soundalike.ml.fulltrack_fusion import FusionConfig, build_nonneg_linear, load_fusion_artifact, save_fusion_artifact
from soundalike.ml.fulltrack_store import (
    FullTrackStore,
    FullTrackStoreReader,
    TrackArtifacts,
    stable_json_sha256,
)
from soundalike.ml.jamendo_fulltrack import ArtistFold, JamendoContext, JamendoTrack, TrackLicense


HASH = hashlib.sha256(b"fulltrack-train-fixture").hexdigest()
CONFIG = hashlib.sha256(b"config").hexdigest()
MODEL = hashlib.sha256(b"model").hexdigest()


class PoisonTags:
    def __iter__(self):
        raise AssertionError("tags must not be iterated")

    def __len__(self):
        raise AssertionError("tags must not be measured")

    def __bool__(self):
        raise AssertionError("tags must not be inspected")


class PoisonTrackTags(dict):
    def __getitem__(self, key):
        raise AssertionError("fold.track_tags must not be read")

    def get(self, key, default=None):
        raise AssertionError("fold.track_tags must not be read")

    def items(self):
        raise AssertionError("fold.track_tags must not be read")


def _unit(index: int, dim: int = 4) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index % dim] = 1.0
    return value


def _license(track_id: int) -> TrackLicense:
    return TrackLicense(
        path=f"track_{track_id}.mp3",
        attribution=f"Track {track_id}",
        name="CC BY",
        url="https://creativecommons.org/licenses/by/4.0/",
        permits_commercial_use=True,
        permits_derivatives=True,
    )


def _track(track_id: int, artist_id: int, *, tags=("genre---rock",)) -> JamendoTrack:
    return JamendoTrack(
        row_index=track_id,
        track_id=track_id,
        artist_id=artist_id,
        album_id=artist_id * 10,
        relative_path=f"track_{track_id}.mp3",
        audio_path=Path(f"/poison/audio/track_{track_id}.mp3"),
        duration_seconds=30.0,
        tags=tags,
        title=f"Track {track_id}",
        artist_name=f"Artist {artist_id}",
        album_name=f"Album {artist_id}",
        release_date="2020-01-01",
        jamendo_url=f"https://www.jamendo.com/track/{track_id}",
        license=_license(track_id),
        expected_audio_sha256=HASH,
        expected_audio_bytes=123,
        fold_parts=(),
    )


def _context(*, poison_tags: bool = False, fold_count: int = 5) -> JamendoContext:
    specs = [
        (100, 1, "train"),
        (101, 1, "train"),
        (102, 2, "train"),
        (103, 3, "train"),
        (104, 4, "validation"),
        (105, 5, "validation"),
        (106, 6, "test"),
        (107, 7, "test"),
    ]
    tags = PoisonTags() if poison_tags else ("genre---rock",)
    tracks = tuple(
        replace(
            _track(track_id, artist_id, tags=tags),
            fold_parts=tuple(part for _ in range(fold_count)),
        )
        for track_id, artist_id, part in specs
    )
    track_parts = {track_id: part for track_id, _, part in specs}
    artist_parts = {artist_id: part for _, artist_id, part in specs}
    folds = []
    for index in range(fold_count):
        folds.append(
            ArtistFold(
                index=index,
                track_parts=MappingProxyType(dict(track_parts)),
                artist_parts=MappingProxyType(dict(artist_parts)),
                track_tags=PoisonTrackTags() if poison_tags else MappingProxyType({tid: ("genre---rock",) for tid, _, _ in specs}),
                tags=("genre---rock",),
            )
        )
    return JamendoContext(
        tracks=tracks,
        folds=tuple(folds),
        metadata_root=Path("/metadata"),
        audio_root=Path("/audio"),
        state_root=Path("/state"),
        metadata_commit="fixture",
        archive_manifest_sha256=HASH,
        track_manifest_sha256=HASH,
        metadata_hashes=MappingProxyType({}),
        source_fingerprint=HASH,
    )


def _artifacts(track_id: int, *, windows: int = 8, dim: int = 4) -> TrackArtifacts:
    base = _unit(track_id, dim)
    alt = _unit(track_id + 1, dim)
    rows = []
    for index in range(windows):
        vec = base + (0.15 * ((index % 3) + 1)) * alt
        vec = vec / np.linalg.norm(vec)
        rows.append(vec.astype(np.float32))
    window_matrix = np.stack(rows)
    return TrackArtifacts(
        global_embedding=(np.mean(window_matrix, axis=0) / np.linalg.norm(np.mean(window_matrix, axis=0))).astype(np.float32),
        window_embeddings=window_matrix,
        window_starts=np.arange(windows, dtype=np.int64) * 10,
        repeated_sections=window_matrix[: min(32, windows)],
        salient_sections=window_matrix[: min(32, windows)],
        repeated_indices=np.arange(min(32, windows), dtype=np.int64),
        salient_indices=np.arange(min(32, windows), dtype=np.int64),
        decoded_samples=windows * 10,
    )


def _sealed_store(tmp_path: Path, context: JamendoContext, *, windows_by_track=None) -> Path:
    root = tmp_path / "store"
    track_ids = [track.track_id for track in context.tracks]
    with FullTrackStore(
        root,
        track_ids=track_ids,
        source_hashes=[HASH] * len(track_ids),
        source_fingerprint=HASH,
        config_sha256=CONFIG,
        model_sha256=MODEL,
        model_id="fixture-model",
        embedding_dim=4,
        shard_tracks=3,
    ) as store:
        for track in context.tracks:
            windows = int(windows_by_track.get(track.track_id, 8)) if windows_by_track else 8
            store.write_track(track.track_id, HASH, _artifacts(track.track_id, windows=windows))
        store.seal()
    return root


class TrackingReader:
    def __init__(self, reader):
        self._reader = reader
        self.binding = reader.binding
        self.manifest = reader.manifest
        self.track_ids = reader.track_ids
        self.read_ids = []

    @property
    def storage_bytes(self):
        return self._reader.storage_bytes

    def read_track(self, track_id):
        self.read_ids.append(int(track_id))
        return self._reader.read_track(track_id)


class CorruptingReader(TrackingReader):
    def __init__(self, reader, *, corrupt_track_id: int, mode: str):
        super().__init__(reader)
        self.corrupt_track_id = int(corrupt_track_id)
        self.mode = mode

    def read_track(self, track_id):
        stored = super().read_track(track_id)
        if int(track_id) != self.corrupt_track_id:
            return stored
        if self.mode == "nonfinite":
            windows = np.asarray(stored.window_embeddings).copy()
            windows[0, 0] = np.nan
            return replace(stored, window_embeddings=windows)
        if self.mode == "misaligned":
            return replace(stored, repeated_indices=np.asarray(stored.repeated_indices[:-1], dtype=np.int64))
        raise AssertionError(f"unknown corrupt mode {self.mode}")


def _training_config(**kwargs) -> fulltrack_train.TrainingConfig:
    params = dict(
        max_epochs=3,
        patience=1,
        hard_negatives=1,
        random_negatives=0,
        min_train_tracks=2,
        min_validation_tracks=2,
        max_train_tracks=4,
        max_validation_tracks=2,
        non_production=True,
        device="cpu",
    )
    params.update(kwargs)
    return fulltrack_train.TrainingConfig(**params)


def test_split_validation_never_returns_or_reads_test_ids(tmp_path):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as base_reader:
        reader = TrackingReader(base_reader)
        splits = fulltrack_train.validate_official_artist_splits(context, reader)
        split = splits[0]
        assert split.test_track_count == 2
        assert "test_track_ids" not in split.as_dict()
        cfg = _training_config()
        train_ids = split.train_track_ids[: cfg.max_train_tracks]
        validation_ids = split.validation_track_ids[: cfg.max_validation_tracks]
        fulltrack_train.build_view_dataset(context, reader, train_ids, fold_index=0, part="train", seed=11)
        fulltrack_train.build_view_dataset(context, reader, validation_ids, fold_index=0, part="validation", seed=11)
        assert set(reader.read_ids).isdisjoint({106, 107})


def test_build_view_dataset_rejects_supplied_test_id_as_train_without_read(tmp_path):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as base_reader:
        reader = TrackingReader(base_reader)
        with pytest.raises(fulltrack_train.FullTrackTrainingError, match="officially assigned.*test.*train"):
            fulltrack_train.build_view_dataset(context, reader, [106], fold_index=0, part="train", seed=11)
        assert reader.read_ids == []


def _context_with_fold(context: JamendoContext, fold_index: int, **updates) -> JamendoContext:
    folds = list(context.folds)
    folds[fold_index] = replace(folds[fold_index], **updates)
    return replace(context, folds=tuple(folds))


@pytest.mark.parametrize(
    ("drift", "expected"),
    [("missing", "missing"), ("extra", "extra")],
)
def test_official_split_requires_exact_track_part_coverage_without_tags(drift, expected):
    context = _context(poison_tags=True)
    track_parts = dict(context.folds[0].track_parts)
    if drift == "missing":
        track_parts.pop(107)
    else:
        track_parts[999] = "test"
    drifted = _context_with_fold(context, 0, track_parts=MappingProxyType(track_parts))
    with pytest.raises(fulltrack_train.FullTrackTrainingError, match=f"track_parts coverage.*{expected}"):
        fulltrack_train.validate_official_artist_splits(
            drifted,
            required_folds=(0,),
            require_all_official=False,
        )


def test_official_split_allows_context_tracks_outside_official_subset():
    context = _context(poison_tags=True)
    unassigned = replace(
        _track(108, 8, tags=PoisonTags()),
        fold_parts=tuple(None for _ in context.folds),
    )
    context = replace(context, tracks=context.tracks + (unassigned,))

    splits = fulltrack_train.validate_official_artist_splits(context)

    assert len(splits) == len(context.folds)
    assert all(split.train_track_count == 4 for split in splits)


@pytest.mark.parametrize(
    ("drift", "expected"),
    [("missing", "missing"), ("extra", "extra")],
)
def test_official_split_requires_exact_artist_part_coverage_without_tags(drift, expected):
    context = _context(poison_tags=True)
    artist_parts = dict(context.folds[0].artist_parts)
    if drift == "missing":
        artist_parts.pop(7)
    else:
        artist_parts[999] = "test"
    drifted = _context_with_fold(context, 0, artist_parts=MappingProxyType(artist_parts))
    with pytest.raises(fulltrack_train.FullTrackTrainingError, match=f"artist_parts coverage.*{expected}"):
        fulltrack_train.validate_official_artist_splits(
            drifted,
            required_folds=(0,),
            require_all_official=False,
        )


def test_store_context_binding_rejects_track_id_coverage_or_order_drift_without_reads(tmp_path):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as base_reader:
        for drifted_ids in (base_reader.track_ids[:-1], tuple(reversed(base_reader.track_ids))):
            reader = TrackingReader(base_reader)
            reader.track_ids = drifted_ids
            with pytest.raises(fulltrack_train.FullTrackTrainingError, match="track IDs.*exactly match"):
                fulltrack_train.validate_store_context_binding(context, reader)
            assert reader.read_ids == []


def test_store_context_binding_rejects_source_hash_plan_drift_without_reads(tmp_path):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    drift_hash = hashlib.sha256(b"drifted-source").hexdigest()
    drifted_tracks = (replace(context.tracks[0], expected_audio_sha256=drift_hash),) + context.tracks[1:]
    drifted_context = replace(context, tracks=drifted_tracks)
    with FullTrackStoreReader(store_root) as base_reader:
        reader = TrackingReader(base_reader)
        with pytest.raises(fulltrack_train.FullTrackTrainingError, match="track plan checksum"):
            fulltrack_train.validate_store_context_binding(drifted_context, reader)
        assert reader.read_ids == []


def test_poison_tags_prove_no_tag_supervision(tmp_path):
    context = _context(poison_tags=True)
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as reader:
        splits = fulltrack_train.validate_official_artist_splits(context, reader)
        dataset = fulltrack_train.build_view_dataset(
            context,
            reader,
            splits[0].train_track_ids[:3],
            fold_index=0,
            part="train",
            seed=3,
        )
    assert dataset.stats["no_tag_supervision"] is True


def test_view_source_time_disjointness_seed_determinism_and_diversity(tmp_path):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as reader:
        stored = reader.read_track(100)
        pair_a = fulltrack_train.make_disjoint_temporal_views(stored, artist_id=1, seed=1, fold_index=0)
        pair_b = fulltrack_train.make_disjoint_temporal_views(stored, artist_id=1, seed=1, fold_index=0)
        pair_c = fulltrack_train.make_disjoint_temporal_views(stored, artist_id=1, seed=2, fold_index=0)
    assert pair_a.pair_hash == pair_b.pair_hash
    assert pair_a.pair_hash != pair_c.pair_hash
    assert set(pair_a.view_a.source_indices).isdisjoint(pair_a.view_b.source_indices)
    assert set(pair_a.view_a.time_starts).isdisjoint(pair_a.view_b.time_starts)
    assert pair_a.source_overlap_count == 0
    assert pair_a.time_overlap_count == 0
    for view in (pair_a.view_a, pair_a.view_b):
        assert np.array_equal(view.repeated_sections, view.window_embeddings[view.repeated_indices])
        assert np.array_equal(view.salient_sections, view.window_embeddings[view.salient_indices])


@pytest.mark.parametrize("row_count", [2, 3, 17, 257])
def test_linear_pairwise_cosine_matches_quadratic_definition(row_count):
    rng = np.random.default_rng(10_000 + row_count)
    matrix = rng.normal(size=(row_count, 64))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    similarities = matrix @ matrix.T
    expected = float(np.mean(similarities[np.triu_indices(row_count, k=1)]))

    actual = fulltrack_train._mean_pairwise_cosine_linear(matrix)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        np.eye(4, dtype=np.float64),
        np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
    ],
)
def test_linear_pairwise_cosine_matches_adversarial_geometries(matrix):
    similarities = matrix @ matrix.T
    expected = float(np.mean(similarities[np.triu_indices(len(matrix), k=1)]))

    actual = fulltrack_train._mean_pairwise_cosine_linear(matrix)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_linear_pairwise_cosine_singleton_matches_existing_definition():
    assert fulltrack_train._mean_pairwise_cosine_linear(np.asarray([[1.0, 0.0]])) == 1.0


def test_track_with_overlapping_single_time_start_is_rejected():
    stored = type(
        "Stored",
        (),
        {
            "track_id": 1,
            "global_embedding": _unit(0),
            "window_embeddings": np.stack([_unit(0), _unit(1)]),
            "window_starts": np.asarray([0, 0], dtype=np.int64),
            "repeated_sections": np.stack([_unit(0), _unit(1)]),
            "salient_sections": np.stack([_unit(0), _unit(1)]),
            "repeated_indices": np.asarray([0, 1], dtype=np.int64),
            "salient_indices": np.asarray([0, 1], dtype=np.int64),
        },
    )()
    with pytest.raises(fulltrack_train.FullTrackTrainingError, match="non-overlapping temporal views"):
        fulltrack_train.make_disjoint_temporal_views(stored, artist_id=1, seed=1)


def test_linear_v2_dataset_hash_excludes_diagnostic_statistics(tmp_path, monkeypatch):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as reader:
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        kwargs = {
            "fold_index": 0,
            "part": "train",
            "seed": 7,
            "min_tracks": 2,
        }
        legacy_default = fulltrack_train.build_view_dataset(
            context,
            reader,
            split.train_track_ids,
            **kwargs,
        )
        legacy_explicit = fulltrack_train.build_view_dataset(
            context,
            reader,
            split.train_track_ids,
            pairwise_cosine_mode="legacy-v1",
            **kwargs,
        )
        linear = fulltrack_train.build_view_dataset(
            context,
            reader,
            split.train_track_ids,
            pairwise_cosine_mode="linear-v2",
            **kwargs,
        )
        original_linear_mean = linear.stats["diversity_pairwise_cosine_mean"]
        monkeypatch.setattr(
            fulltrack_train,
            "_mean_pairwise_cosine_linear",
            lambda matrix: float(original_linear_mean) + 0.125,
        )
        altered_diagnostic = fulltrack_train.build_view_dataset(
            context,
            reader,
            split.train_track_ids,
            pairwise_cosine_mode="linear-v2",
            **kwargs,
        )

    assert legacy_default.dataset_hash == legacy_explicit.dataset_hash
    assert [pair.pair_hash for pair in legacy_default.pairs] == [
        pair.pair_hash for pair in linear.pairs
    ]
    assert linear.stats["diversity_pairwise_cosine_mean"] == pytest.approx(
        legacy_default.stats["diversity_pairwise_cosine_mean"],
        rel=1e-12,
        abs=1e-12,
    )
    assert "diversity_pairwise_cosine_algorithm" not in legacy_default.stats
    assert linear.stats["diversity_pairwise_cosine_algorithm"] == "sum-vector-v2"
    assert linear.dataset_hash != legacy_default.dataset_hash
    assert altered_diagnostic.dataset_hash == linear.dataset_hash
    assert (
        altered_diagnostic.stats["diversity_pairwise_cosine_mean"]
        != linear.stats["diversity_pairwise_cosine_mean"]
    )


def test_build_view_dataset_counts_valid_unsplittable_tracks_as_rejections(tmp_path):
    context = _context()
    store_root = _sealed_store(tmp_path, context, windows_by_track={100: 1})
    with FullTrackStoreReader(store_root) as reader:
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        dataset = fulltrack_train.build_view_dataset(
            context,
            reader,
            split.train_track_ids[:3],
            fold_index=0,
            part="train",
            seed=7,
            min_tracks=2,
        )
    assert dataset.rejected_track_count == 1
    assert dataset.rejected_reasons == {"fewer_than_two_windows": 1}
    assert 100 not in dataset.track_ids


@pytest.mark.parametrize(
    ("mode", "message"),
    [("nonfinite", "non-finite"), ("misaligned", "misaligned")],
)
def test_build_view_dataset_propagates_store_integrity_failures(tmp_path, mode, message):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as base_reader:
        reader = CorruptingReader(base_reader, corrupt_track_id=100, mode=mode)
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        with pytest.raises(fulltrack_train.FullTrackTrainingError, match=message):
            fulltrack_train.build_view_dataset(
                context,
                reader,
                split.train_track_ids[:3],
                fold_index=0,
                part="train",
                seed=7,
                min_tracks=2,
            )


def test_hard_negative_mining_excludes_same_artist(tmp_path):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as reader:
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        dataset = fulltrack_train.build_view_dataset(
            context,
            reader,
            split.train_track_ids,
            fold_index=0,
            part="train",
            seed=7,
        )
    ranking = fulltrack_train.mine_negatives(dataset, config=_training_config(max_train_tracks=None), seed=7)
    artist_by_track = {track.track_id: track.artist_id for track in context.tracks}
    for query_track, neg_artists in zip(ranking.query_track_ids, ranking.negative_artist_ids):
        assert all(artist != artist_by_track[query_track] for artist in neg_artists)
    assert ranking.stats["false_negative_safeguards"]["same_artist_negatives_excluded"] is True


def _simple_view(track_id: int, artist_id: int, side: int) -> fulltrack_train.TemporalView:
    windows = np.stack([_unit(0), _unit(0)]).astype(np.float32)
    payload = {"track_id": track_id, "artist_id": artist_id, "side": side}
    return fulltrack_train.TemporalView(
        track_id=track_id,
        artist_id=artist_id,
        view_index=side,
        source_indices=(side,),
        time_starts=(side * 10,),
        global_embedding=_unit(0),
        window_embeddings=windows,
        window_starts=np.asarray([0, 10], dtype=np.int64),
        repeated_sections=np.zeros((0, 4), dtype=np.float32),
        salient_sections=np.zeros((0, 4), dtype=np.float32),
        repeated_indices=np.zeros((0,), dtype=np.int64),
        salient_indices=np.zeros((0,), dtype=np.int64),
        view_hash=fulltrack_train.stable_json_sha256(payload),
    )


def _simple_dataset(track_specs=((1, 10), (2, 20), (3, 30))) -> fulltrack_train.ViewDataset:
    pairs = []
    for track_id, artist_id in track_specs:
        a = _simple_view(track_id, artist_id, 0)
        b = _simple_view(track_id, artist_id, 1)
        pairs.append(
            fulltrack_train.ViewPair(
                track_id=track_id,
                artist_id=artist_id,
                view_a=a,
                view_b=b,
                track_window_count=2,
                source_overlap_count=0,
                time_overlap_count=0,
                source_coverage=1.0,
                positive_cosine=1.0,
                pair_hash=fulltrack_train.stable_json_sha256({"track": track_id}),
            )
        )
    stats = {"track_count": len(pairs), "overlap": 0, "no_tag_supervision": True}
    return fulltrack_train.ViewDataset(
        fold_index=0,
        part="train",
        seed=1,
        pairs=tuple(pairs),
        rejected_track_count=0,
        rejected_reasons={},
        dataset_hash=fulltrack_train.stable_json_sha256({"pairs": [p.pair_hash for p in pairs]}),
        stats=stats,
        embedding_dim=4,
    )


def test_v2_diagnostic_changes_do_not_change_negative_mining():
    dataset = _simple_dataset(((1, 10), (2, 20), (3, 30), (4, 40)))
    altered_stats = dict(dataset.stats)
    altered_stats["diversity_pairwise_cosine_mean"] = 0.987654321
    altered_stats["diversity_score"] = 0.0061728395
    altered = replace(dataset, stats=altered_stats)
    cfg = _training_config(
        hard_negatives=2,
        random_negatives=2,
        max_train_tracks=None,
        max_validation_tracks=None,
    )

    original_ranking = fulltrack_train.mine_negatives(dataset, config=cfg, seed=5)
    altered_ranking = fulltrack_train.mine_negatives(altered, config=cfg, seed=5)

    assert altered.dataset_hash == dataset.dataset_hash
    assert altered_ranking.ranking_hash == original_ranking.ranking_hash
    assert altered_ranking.negative_track_ids == original_ranking.negative_track_ids
    assert np.array_equal(altered_ranking.negative_indices, original_ranking.negative_indices)
    assert np.array_equal(altered_ranking.pos_features, original_ranking.pos_features)
    assert np.array_equal(altered_ranking.neg_features, original_ranking.neg_features)


def test_deterministic_hard_negative_tie_handling():
    dataset = _simple_dataset()
    cfg = _training_config(hard_negatives=2, random_negatives=0, max_train_tracks=None, max_validation_tracks=None)
    ranking = fulltrack_train.mine_negatives(dataset, config=cfg, seed=5)
    assert ranking.negative_track_ids[0] == (2, 3)
    assert ranking.negative_track_ids[1] == (2, 3)


def test_random_negatives_consume_remaining_candidates_before_replacement():
    dataset = _simple_dataset(((1, 10), (2, 20), (3, 30), (4, 40)))
    cfg = _training_config(hard_negatives=2, random_negatives=2, max_train_tracks=None, max_validation_tracks=None)
    ranking = fulltrack_train.mine_negatives(dataset, config=cfg, seed=0)
    assert ranking.negative_track_ids[0][:2] == (2, 3)
    assert ranking.hard_negative_mask[0].tolist() == [True, True, False, False]
    assert 4 in ranking.negative_track_ids[0][2:]


def test_three_seed_train_all_plan_has_45_jobs():
    plan = fulltrack_train.build_train_all_plan([1, 2, 3])
    assert plan.job_count == 45
    assert {job.fold_index for job in plan.jobs} == {0, 1, 2, 3, 4}
    assert set(plan.candidates) == set(fulltrack_train.CANDIDATE_KINDS)
    with pytest.raises(fulltrack_train.FullTrackTrainingError):
        fulltrack_train.build_train_all_plan([1, 2])


def test_train_all_prepares_once_per_fold_seed_and_preserves_plan_order(
    tmp_path, monkeypatch
):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    prepared_by_group = {}
    run_order = []

    def fake_prepare(
        context_arg,
        reader_arg,
        split,
        *,
        seed,
        config,
        pairwise_cosine_mode="legacy-v1",
    ):
        assert pairwise_cosine_mode == "linear-v2"
        key = (split.fold_index, seed)
        prepared_by_group[key] = object()
        return prepared_by_group[key]

    def fake_run(
        context_arg,
        reader_arg,
        split,
        spec,
        *,
        config,
        output_dir,
        prepared_data=None,
        dedicated_cuda_stream=False,
    ):
        key = (spec.fold_index, spec.seed)
        assert prepared_data is prepared_by_group[key]
        assert dedicated_cuda_stream
        run_order.append(spec.job_id)
        return fulltrack_train.JobRunResult(
            spec=spec,
            job_dir=Path(output_dir) / spec.relative_dir,
            status="trained",
            report={"report_sha256": HASH},
        )

    monkeypatch.setattr(fulltrack_train, "prepare_training_data", fake_prepare)
    monkeypatch.setattr(fulltrack_train, "run_train_job", fake_run)
    with FullTrackStoreReader(store_root) as reader:
        result = fulltrack_train.train_all(
            context,
            reader,
            output_dir=tmp_path / "out",
            seeds=(17, 29, 43),
            config=_training_config(),
            candidate_workers=2,
            pairwise_cosine_mode="linear-v2",
        )

    assert result["pairwise_cosine_mode"] == "linear-v2"
    plan = fulltrack_train.build_train_all_plan((17, 29, 43))
    assert len(prepared_by_group) == 15
    assert len(run_order) == 45
    assert [item["job_id"] for item in result["results"]] == [
        spec.job_id for spec in plan.jobs
    ]


def test_reusable_job_recovers_only_empty_interrupted_directory(tmp_path):
    spec = fulltrack_train.build_train_all_plan([17, 29, 43]).jobs[0]
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    result = fulltrack_train.try_load_reusable_job(
        job_dir,
        spec=spec,
        training_config_sha256=HASH,
        source_fingerprint=HASH,
        store_binding_hash=HASH,
        store_manifest_sha256=HASH,
    )

    assert result is None
    assert not job_dir.exists()

    job_dir.mkdir()
    (job_dir / "unexpected.bin").write_bytes(b"unreported")
    with pytest.raises(fulltrack_train.FullTrackTrainingError, match="stale/incomplete"):
        fulltrack_train.try_load_reusable_job(
            job_dir,
            spec=spec,
            training_config_sha256=HASH,
            source_fingerprint=HASH,
            store_binding_hash=HASH,
            store_manifest_sha256=HASH,
        )
    assert (job_dir / "unexpected.bin").read_bytes() == b"unreported"


def test_production_config_rejects_lightweight_epoch_and_track_limit_overrides():
    fulltrack_train.TrainingConfig().validate()
    fulltrack_train.TrainingConfig(
        max_epochs=3,
        max_train_tracks=2,
        max_validation_tracks=2,
        non_production=True,
    ).validate()
    with pytest.raises(fulltrack_train.FullTrackTrainingError, match="max_epochs.*non-production"):
        fulltrack_train.TrainingConfig(max_epochs=3).validate()
    with pytest.raises(fulltrack_train.FullTrackTrainingError, match="track limits.*non-production"):
        fulltrack_train.TrainingConfig(max_train_tracks=2).validate()
    with pytest.raises(fulltrack_train.FullTrackTrainingError, match="track limits.*non-production"):
        fulltrack_train.TrainingConfig(max_validation_tracks=2).validate()


def test_train_all_and_run_train_job_apply_production_config_gating(tmp_path):
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    production_lightweight = fulltrack_train.TrainingConfig(max_epochs=3, device="cpu")
    with FullTrackStoreReader(store_root) as reader:
        with pytest.raises(fulltrack_train.FullTrackTrainingError, match="max_epochs.*non-production"):
            fulltrack_train.train_all(
                context,
                reader,
                output_dir=tmp_path / "out-train-all",
                config=production_lightweight,
                dry_run=True,
            )
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        spec = fulltrack_train.TrainJobSpec(
            fold_index=0,
            candidate_kind="nonnegative_linear",
            seed=31,
            job_id="fold-0__nonnegative_linear__seed-31",
            relative_dir="fold-0/nonnegative_linear/seed-31",
        )
        with pytest.raises(fulltrack_train.FullTrackTrainingError, match="max_epochs.*non-production"):
            fulltrack_train.run_train_job(
                context,
                reader,
                split,
                spec,
                config=production_lightweight,
                output_dir=tmp_path / "out-run-job",
            )


def test_run_train_job_production_fails_on_rejected_official_train_track(tmp_path):
    context = _context()
    store_root = _sealed_store(tmp_path, context, windows_by_track={100: 1})
    with FullTrackStoreReader(store_root) as reader:
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        spec = fulltrack_train.TrainJobSpec(
            fold_index=0,
            candidate_kind="nonnegative_linear",
            seed=31,
            job_id="fold-0__nonnegative_linear__seed-31",
            relative_dir="fold-0/nonnegative_linear/seed-31",
        )
        with pytest.raises(fulltrack_train.FullTrackTrainingError, match="production training rejected 1 official train"):
            fulltrack_train.run_train_job(
                context,
                reader,
                split,
                spec,
                config=fulltrack_train.TrainingConfig(device="cpu"),
                output_dir=tmp_path / "out",
            )


def test_checkpoint_safe_loading_and_tamper_rejection(tmp_path):
    arrays = {"weights": np.ones(fulltrack_train.FEATURE_DIM, dtype=np.float64)}
    checkpoint = fulltrack_train.save_training_checkpoint(
        tmp_path / "checkpoint",
        job_id="job",
        kind="nonnegative_linear",
        fold=0,
        seed=1,
        hidden_dims=(),
        embedding_dim=4,
        training_config_sha256=HASH,
        job_config_sha256=HASH,
        source_fingerprint=HASH,
        store_binding_sha256=HASH,
        arrays=arrays,
    )
    loaded = fulltrack_train.load_training_checkpoint(tmp_path / "checkpoint", expected_kind="nonnegative_linear")
    assert np.allclose(loaded.arrays["weights"], arrays["weights"])
    assert loaded.metadata["checkpoint_sha256"] == checkpoint.metadata["checkpoint_sha256"]
    (tmp_path / "checkpoint" / "checkpoint.npz").write_bytes(b"tampered")
    with pytest.raises(fulltrack_train.FullTrackTrainingError, match="checksum"):
        fulltrack_train.load_training_checkpoint(tmp_path / "checkpoint", expected_kind="nonnegative_linear")


def test_checkpoint_load_parses_the_pinned_hashed_npz_snapshot(tmp_path, monkeypatch):
    arrays = {"weights": np.ones(fulltrack_train.FEATURE_DIM, dtype=np.float64)}
    fulltrack_train.save_training_checkpoint(
        tmp_path / "checkpoint",
        job_id="job",
        kind="nonnegative_linear",
        fold=0,
        seed=1,
        hidden_dims=(),
        embedding_dim=4,
        training_config_sha256=HASH,
        job_config_sha256=HASH,
        source_fingerprint=HASH,
        store_binding_sha256=HASH,
        arrays=arrays,
    )
    original_npz = (tmp_path / "checkpoint" / "checkpoint.npz").read_bytes()
    real_safe_read = fulltrack_train._safe_read_bytes
    npz_reads = {"count": 0}

    def racing_safe_read(path, label, max_bytes):
        if Path(path).name == "checkpoint.npz":
            npz_reads["count"] += 1
            return original_npz if npz_reads["count"] == 1 else b"tampered"
        return real_safe_read(path, label, max_bytes)

    monkeypatch.setattr(fulltrack_train, "_safe_read_bytes", racing_safe_read)
    loaded = fulltrack_train.load_training_checkpoint(tmp_path / "checkpoint", expected_kind="nonnegative_linear")
    assert npz_reads["count"] == 1
    assert np.allclose(loaded.arrays["weights"], arrays["weights"])


def test_fusion_model_safe_loading_round_trip(tmp_path):
    config = FusionConfig(
        kind="nonnegative_linear",
        embedding_dim=4,
        model_id="fixture",
        store_id=HASH,
        config_sha256=HASH,
        seed=1,
        fold_index=0,
    )
    model = build_nonneg_linear(np.ones(fulltrack_train.FEATURE_DIM, dtype=np.float64), config)
    save_fusion_artifact(model, tmp_path / "model")
    loaded = load_fusion_artifact(tmp_path / "model")
    assert loaded.config.kind == "nonnegative_linear"
    (tmp_path / "model" / "weights.npz").write_bytes(b"tampered")
    with pytest.raises(Exception):
        load_fusion_artifact(tmp_path / "model")


def test_no_audio_decoding_imports_or_audio_access(tmp_path):
    source = Path(fulltrack_train.__file__).read_text(encoding="utf-8")
    for forbidden in ("librosa", "soundfile", "torchaudio", "import av"):
        assert forbidden not in source
    context = _context(poison_tags=True)
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as reader:
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        fulltrack_train.build_view_dataset(
            context,
            reader,
            split.train_track_ids[:2],
            fold_index=0,
            part="train",
            seed=1,
        )


@pytest.mark.parametrize("kind", fulltrack_train.CANDIDATE_KINDS)
def test_tiny_training_deterministic_finite_model_for_each_candidate_when_torch_available(tmp_path, kind):
    pytest.importorskip("torch")
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as reader:
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        cfg = _training_config()
        train_dataset = fulltrack_train.build_view_dataset(
            context,
            reader,
            split.train_track_ids[:4],
            fold_index=0,
            part="train",
            seed=19,
            min_tracks=2,
        )
        validation_dataset = fulltrack_train.build_view_dataset(
            context,
            reader,
            split.validation_track_ids[:2],
            fold_index=0,
            part="validation",
            seed=19,
            min_tracks=2,
        )
    train_ranking = fulltrack_train.mine_negatives(
        train_dataset, config=cfg, seed=120
    )
    validation_ranking = fulltrack_train.mine_negatives(
        validation_dataset, config=cfg, seed=221
    )
    first = fulltrack_train.train_candidate_from_datasets(
        kind,
        train_dataset,
        validation_dataset,
        config=cfg,
        seed=19,
        store_binding_hash=HASH,
        source_fingerprint=HASH,
        job_config_sha256=HASH,
        train_ranking=train_ranking,
        validation_ranking=validation_ranking,
    )
    second = fulltrack_train.train_candidate_from_datasets(
        kind,
        train_dataset,
        validation_dataset,
        config=cfg,
        seed=19,
        store_binding_hash=HASH,
        source_fingerprint=HASH,
        job_config_sha256=HASH,
    )
    for name, array in first.arrays.items():
        assert np.all(np.isfinite(array))
        assert np.allclose(array, second.arrays[name])
    assert first.report["history"] == second.report["history"]
    assert first.train_ranking.ranking_hash == second.train_ranking.ranking_hash
    assert first.validation_ranking.ranking_hash == second.validation_ranking.ranking_hash
    assert first.report["early_stopping_metric"] == "validation_self_supervised_ranking_loss"
    assert first.report["epochs_ran"] <= cfg.max_epochs
    best_row = first.report["history"][first.report["best_epoch"] - 1]
    for metric in (
        "train_loss",
        "validation_loss",
        "train_ranking_accuracy",
        "validation_ranking_accuracy",
        "train_pairwise_auc",
        "validation_pairwise_auc",
    ):
        assert first.report[metric] == pytest.approx(best_row[metric])
    score = first.model.score_candidate(validation_dataset.pairs[0].view_a, validation_dataset.pairs[0].view_b)
    assert np.isfinite(score)


def test_complete_resume_reuse_and_tamper_rejection_when_torch_available(tmp_path):
    pytest.importorskip("torch")
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    with FullTrackStoreReader(store_root) as reader:
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        spec = fulltrack_train.TrainJobSpec(
            fold_index=0,
            candidate_kind="nonnegative_linear",
            seed=31,
            job_id="fold-0__nonnegative_linear__seed-31",
            relative_dir="fold-0/nonnegative_linear/seed-31",
        )
        cfg = _training_config()
        first = fulltrack_train.run_train_job(context, reader, split, spec, config=cfg, output_dir=tmp_path / "out")
        second = fulltrack_train.run_train_job(context, reader, split, spec, config=cfg, output_dir=tmp_path / "out")
        assert first.status == "trained"
        assert second.status == "reused"
        report_path = second.job_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        from soundalike.ml.fulltrack_eval import (
            FullTrackEvaluationError,
            load_trained_model_for_fold,
        )

        load_trained_model_for_fold(
            tmp_path / "out",
            fold_index=0,
            candidate_kind="nonnegative_linear",
            seed=31,
            expected_source_fingerprint=context.source_fingerprint,
            expected_store_binding_sha256=fulltrack_train.store_binding_sha256(reader),
            store_embedding_dim=reader.binding.embedding_dim,
            store_repetition_sections=reader.binding.repetition_sections,
            store_salient_sections=reader.binding.salient_sections,
        )
        checkpoint_dir = second.job_dir / "checkpoint"
        original_checkpoint_json = (checkpoint_dir / "checkpoint.json").read_bytes()
        original_checkpoint_npz = (checkpoint_dir / "checkpoint.npz").read_bytes()
        fulltrack_train.save_training_checkpoint(
            checkpoint_dir,
            job_id=spec.job_id,
            kind=spec.candidate_kind,
            fold=spec.fold_index,
            seed=spec.seed,
            hidden_dims=(),
            embedding_dim=reader.binding.embedding_dim,
            training_config_sha256=cfg.sha256,
            job_config_sha256=report["job_config_sha256"],
            source_fingerprint=context.source_fingerprint,
            store_binding_sha256=fulltrack_train.store_binding_sha256(reader),
            arrays={
                "weights": np.full(
                    fulltrack_train.FEATURE_DIM, 2.0, dtype=np.float64
                )
            },
        )
        with pytest.raises(
            fulltrack_train.FullTrackTrainingError,
            match="checkpoint/report hash binding",
        ):
            fulltrack_train.run_train_job(
                context,
                reader,
                split,
                spec,
                config=cfg,
                output_dir=tmp_path / "out",
            )
        (checkpoint_dir / "checkpoint.json").write_bytes(original_checkpoint_json)
        (checkpoint_dir / "checkpoint.npz").write_bytes(original_checkpoint_npz)
        forged = json.loads(json.dumps(report))
        forged["training_config"]["learning_rate"] *= 2.0
        forged["report_sha256"] = stable_json_sha256(
            {key: value for key, value in forged.items() if key != "report_sha256"}
        )
        report_path.write_text(json.dumps(forged), encoding="utf-8")
        with pytest.raises(FullTrackEvaluationError, match="training config"):
            load_trained_model_for_fold(
                tmp_path / "out",
                fold_index=0,
                candidate_kind="nonnegative_linear",
                seed=31,
                expected_source_fingerprint=context.source_fingerprint,
                expected_store_binding_sha256=fulltrack_train.store_binding_sha256(reader),
                store_embedding_dim=reader.binding.embedding_dim,
                store_repetition_sections=reader.binding.repetition_sections,
                store_salient_sections=reader.binding.salient_sections,
            )
        report_path.write_text(json.dumps(report), encoding="utf-8")
        report["seed"] = 999
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(fulltrack_train.FullTrackTrainingError):
            fulltrack_train.run_train_job(context, reader, split, spec, config=cfg, output_dir=tmp_path / "out")


def test_run_train_job_reuses_preview_rankings_when_torch_available(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    calls = []
    real_mine_negatives = fulltrack_train.mine_negatives

    def tracking_mine_negatives(dataset, *, config=None, seed=0):
        calls.append(int(seed))
        return real_mine_negatives(dataset, config=config, seed=seed)

    monkeypatch.setattr(fulltrack_train, "mine_negatives", tracking_mine_negatives)
    with FullTrackStoreReader(store_root) as reader:
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        spec = fulltrack_train.TrainJobSpec(
            fold_index=0,
            candidate_kind="nonnegative_linear",
            seed=31,
            job_id="fold-0__nonnegative_linear__seed-31",
            relative_dir="fold-0/nonnegative_linear/seed-31",
        )
        fulltrack_train.run_train_job(
            context,
            reader,
            split,
            spec,
            config=_training_config(),
            output_dir=tmp_path / "out",
        )

    assert calls == [132, 233]


def test_concurrent_cuda_streams_match_serial_candidates(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    context = _context()
    store_root = _sealed_store(tmp_path, context)
    cfg = _training_config(device="cuda")
    with FullTrackStoreReader(store_root) as reader:
        split = fulltrack_train.validate_official_artist_splits(context, reader)[0]
        prepared = fulltrack_train.prepare_training_data(
            context,
            reader,
            split,
            seed=31,
            config=cfg,
        )

    def fit(kind, dedicated):
        return fulltrack_train.train_candidate_from_datasets(
            kind,
            prepared.train_dataset,
            prepared.validation_dataset,
            config=cfg,
            seed=31,
            store_binding_hash=HASH,
            source_fingerprint=HASH,
            job_config_sha256=HASH,
            train_ranking=prepared.train_ranking,
            validation_ranking=prepared.validation_ranking,
            dedicated_cuda_stream=dedicated,
        )

    kinds = fulltrack_train.CANDIDATE_KINDS
    serial = {kind: fit(kind, False) for kind in kinds}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {kind: executor.submit(fit, kind, True) for kind in kinds}
        concurrent_results = {
            kind: future.result() for kind, future in futures.items()
        }

    for kind in kinds:
        assert concurrent_results[kind].report["history"] == serial[kind].report["history"]
        for name, array in serial[kind].arrays.items():
            assert np.array_equal(concurrent_results[kind].arrays[name], array)


def test_finite_gradient_failure_helper_when_torch_available():
    torch = pytest.importorskip("torch")
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([float("nan")])
    with pytest.raises(fulltrack_train.FullTrackTrainingError, match="non-finite gradient"):
        fulltrack_train.assert_finite_gradients([param])
