import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from soundalike.ml.fulltrack_v2_diagnostics import (
    CandidateCeilingConfig,
    FullTrackDiagnosticError,
    candidate_ceiling_report,
)
from soundalike.ml.jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    ArtistFold,
    JamendoContext,
    JamendoTrack,
    TrackLicense,
)


HASH = hashlib.sha256(b"v2-diagnostic-fixture").hexdigest()


def _track(index, artist_id, tags):
    relative = f"{index:02d}/{index}.mp3"
    return JamendoTrack(
        row_index=index,
        track_id=100 + index,
        artist_id=artist_id,
        album_id=300 + index,
        relative_path=relative,
        audio_path=Path(relative),
        duration_seconds=60.0,
        tags=tuple(tags),
        title=f"Track {index}",
        artist_name=f"Artist {artist_id}",
        album_name="Fixture",
        release_date="2026",
        jamendo_url=f"https://www.jamendo.com/track/{100 + index}",
        license=TrackLicense(
            path=relative,
            attribution="fixture",
            name="CC",
            url="https://creativecommons.org/licenses/by-nc-sa/3.0/",
            permits_commercial_use=False,
            permits_derivatives=True,
        ),
        expected_audio_sha256=HASH,
        expected_audio_bytes=1,
        fold_parts=("validation",),
    )


def _fixture():
    rock = ("genre---rock", "instrument---guitar")
    jazz = ("genre---jazz", "instrument---piano")
    tracks = (
        _track(0, 200, rock),
        _track(1, 201, jazz),
        _track(2, 202, jazz),
        _track(3, 203, rock),
    )
    fold = ArtistFold(
        index=0,
        track_parts=MappingProxyType({track.track_id: "validation" for track in tracks}),
        artist_parts=MappingProxyType({track.artist_id: "validation" for track in tracks}),
        track_tags=MappingProxyType({track.track_id: track.tags for track in tracks}),
        tags=tuple(sorted(set(rock + jazz))),
    )
    context = JamendoContext(
        tracks=tracks,
        folds=(fold,),
        metadata_root=Path("metadata"),
        audio_root=Path("audio"),
        state_root=Path("state"),
        metadata_commit="fixture",
        archive_manifest_sha256=HASH,
        track_manifest_sha256=HASH,
        metadata_hashes=MappingProxyType({}),
        source_fingerprint=HASH,
        evidence_scope=EVIDENCE_SCOPE,
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.995, 0.1],
            [0.0, 1.0],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )
    binding = {
        "schema_version": 2,
        "source_fingerprint": HASH,
        "config_sha256": HASH,
        "model_sha256": HASH,
        "model_id": "fixture",
        "embedding_dim": 2,
        "track_count": 4,
        "shard_tracks": 2,
        "repetition_sections": 2,
        "salient_sections": 2,
        "track_plan_sha256": HASH,
    }
    reader = SimpleNamespace(
        track_ids=tuple(track.track_id for track in tracks),
        global_embeddings=embeddings,
        binding=SimpleNamespace(as_dict=lambda: dict(binding)),
    )
    return context, reader


def test_candidate_ceiling_matches_known_global_ranks():
    context, reader = _fixture()
    report = candidate_ceiling_report(
        context,
        reader,
        config=CandidateCeilingConfig(
            pool_sizes=(1, 2, 3),
            recall_cutoff=1,
            min_shared_tags=2,
            min_tag_jaccard=1.0,
        ),
    )

    assert report["test_partition_accessed"] is False
    assert report["query_count"] == 4
    assert report["global_recall_at_k"] == pytest.approx(0.0)
    assert report["pools"]["1"]["query_hit_rate"] == pytest.approx(0.0)
    assert report["pools"]["2"]["query_hit_rate"] == pytest.approx(0.75)
    assert report["pools"]["3"]["query_hit_rate"] == pytest.approx(1.0)
    assert report["pools"]["3"]["mean_oracle_recall_at_k"] == pytest.approx(1.0)
    assert len(report["report_payload_sha256"]) == 64


def test_candidate_ceiling_rejects_test_partition():
    context, reader = _fixture()
    with pytest.raises(FullTrackDiagnosticError, match="only train or validation"):
        candidate_ceiling_report(
            context,
            reader,
            config=CandidateCeilingConfig(part="test"),
        )