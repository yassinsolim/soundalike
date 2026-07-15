import hashlib
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.fulltrack_store import (
    FullTrackStore,
    FullTrackStoreError,
    FullTrackStoreReader,
    TrackArtifacts,
)


SOURCE = hashlib.sha256(b"source").hexdigest()
CONFIG = hashlib.sha256(b"config").hexdigest()
MODEL = hashlib.sha256(b"model").hexdigest()


def unit(index: int, dimension: int = 4) -> np.ndarray:
    value = np.zeros(dimension, dtype=np.float32)
    value[index % dimension] = 1.0
    return value


def artifacts(index: int) -> TrackArtifacts:
    windows = np.stack((unit(index), unit(index + 1)))
    return TrackArtifacts(
        global_embedding=unit(index),
        window_embeddings=windows,
        window_starts=np.asarray([0, 5], dtype=np.int64),
        repeated_sections=windows,
        salient_sections=windows,
        repeated_indices=np.asarray([0, 1], dtype=np.int64),
        salient_indices=np.asarray([0, 1], dtype=np.int64),
        decoded_samples=15,
    )


def open_store(root: Path, *, config: str = CONFIG) -> FullTrackStore:
    return FullTrackStore(
        root,
        track_ids=(10, 11, 12),
        source_hashes=(SOURCE, SOURCE, SOURCE),
        source_fingerprint=SOURCE,
        config_sha256=config,
        model_sha256=MODEL,
        model_id="fake-model",
        embedding_dim=4,
        shard_tracks=2,
    )


def finish_store(root: Path) -> None:
    with open_store(root) as store:
        for index, track_id in enumerate((10, 11, 12)):
            if track_id in store.pending_track_ids():
                store.write_track(track_id, SOURCE, artifacts(index))
        store.seal()


def test_store_resumes_partial_fixed_range_and_reads_variable_arrays(tmp_path):
    root = tmp_path / "store"
    store = open_store(root)
    store.write_track(10, SOURCE, artifacts(0))
    store.close(flush=True)
    resumed = open_store(root)
    assert resumed.completed_count == 1
    resumed.close(flush=False)

    finish_store(root)
    with FullTrackStoreReader(
        root,
        expected_source_fingerprint=SOURCE,
        expected_config_sha256=CONFIG,
        expected_model_sha256=MODEL,
    ) as reader:
        value = reader.read_track(11)
        assert value.window_embeddings.shape == (2, 4)
        assert value.repeated_sections.shape == (2, 4)
        assert value.salient_sections.shape == (2, 4)
        assert value.repeated_indices.tolist() == [0, 1]
        assert value.salient_indices.tolist() == [0, 1]
        assert reader.binding.repetition_sections == 32
        assert reader.binding.salient_sections == 32
        assert reader.manifest["schema_version"] == 2
        assert reader.track_ids == (10, 11, 12)
        assert reader.storage_bytes > 0


def test_unsealed_interruption_reprocesses_instead_of_silently_skipping(tmp_path):
    root = tmp_path / "store"
    store = open_store(root)
    store.write_track(10, SOURCE, artifacts(0))
    store.close(flush=False)
    resumed = open_store(root)
    assert resumed.pending_track_ids() == (10, 11, 12)
    resumed.write_track(10, SOURCE, artifacts(0))
    with pytest.raises(FullTrackStoreError, match="already exists"):
        resumed.write_track(10, SOURCE, artifacts(0))
    resumed.close(flush=False)


def test_binding_and_source_drift_fail_closed(tmp_path):
    root = tmp_path / "store"
    store = open_store(root)
    with pytest.raises(FullTrackStoreError, match="source hash drift"):
        store.write_track(10, "f" * 64, artifacts(0))
    store.close(flush=False)
    with pytest.raises(FullTrackStoreError, match="binding drift"):
        open_store(root, config="e" * 64)


def test_store_rejects_false_section_source_window_provenance(tmp_path):
    root = tmp_path / "store"
    store = open_store(root)
    dishonest = replace(
        artifacts(0),
        repeated_indices=np.asarray([1, 0], dtype=np.int64),
    )
    with pytest.raises(FullTrackStoreError, match="do not match"):
        store.write_track(10, SOURCE, dishonest)
    store.close(flush=False)


def test_store_allows_only_one_writer(tmp_path):
    root = tmp_path / "store"
    first = open_store(root)
    with pytest.raises(FullTrackStoreError, match="another writer"):
        open_store(root)
    first.close(flush=False)
    second = open_store(root)
    second.close(flush=False)


def test_global_corruption_is_detected_on_restart(tmp_path):
    root = tmp_path / "store"
    store = open_store(root)
    store.write_track(10, SOURCE, artifacts(0))
    store.close(flush=True)
    with (root / "global.f16").open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00\x00")
    with pytest.raises(FullTrackStoreError, match="global embedding corruption"):
        open_store(root)


def test_shard_corruption_and_sealed_checksum_are_detected(tmp_path):
    root = tmp_path / "store"
    finish_store(root)
    shard = next((root / "shards").glob("*.npz"))
    with shard.open("r+b") as handle:
        handle.seek(10)
        handle.write(b"corrupt")
    with pytest.raises(FullTrackStoreError, match="shard checksum"):
        FullTrackStoreReader(root)


def test_reader_rejects_ledger_routing_not_bound_by_sealed_manifest(tmp_path):
    root = tmp_path / "store-routing"
    finish_store(root)
    original = sorted((root / "shards").glob("*.npz"))[0]
    rogue = original.with_name("rogue.npz")
    shutil.copy2(original, rogue)
    with sqlite3.connect(root / "ledger.sqlite3") as connection:
        connection.execute(
            "UPDATE shards SET file_name=? WHERE shard_start=0",
            (rogue.name,),
        )
    with pytest.raises(FullTrackStoreError, match="manifest/ledger shard routing"):
        FullTrackStoreReader(root)
