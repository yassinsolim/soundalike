"""Unit tests for soundalike.ml.audio_experiments (iteration-4 pipeline).

All tests run without real data, long training, GPU, or network access.
Torch-dependent tests are skipped if torch is not installed.

Test groups:
  A. ExperimentConfig — defaults, JSON round-trip, hash stability.
  B. CheckpointMeta — save / load, field round-trip.
  C. ResourceLog — field access and serialisation.
  D. LeakageGuard — v5 benchmark loading, artist extraction, exclusion mask.
  E. _stratified_split — stratification correctness.
  F. FMAPackedDataset — shapes, dtypes, augmentation, split sizes.
  G. SupConLoss — same-label lower loss, cross-artist masking, edge cases.
  H. BYOL components — EMA update, byol_loss, model forward.
  I. DistillationDataset — exclusion, shapes, item types.
  J. distillation_loss — finite, gradient flows, direction.
  K. CatalogExtractor — float16 output, batch correctness.
  L. V5PairResolver — title/artist matching, derivative penalty.
  M. DevEvaluator — all metric formulae, primary, candidate recall, bootstrap.
  N. LateInteractionReranker — MaxSim formula, rerank ordering.
  O. AudioExperimentPipeline — smoke tests (evaluate_dev with synthetic data).
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional torch import — skip GPU tests gracefully.
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")  # type: ignore[assignment]

from soundalike.ml.audio_experiments import (
    _build_teacher,
    _credited_artists,
    _normalize,
    _primary_artist,
    _stratified_split,
    AudioExperimentPipeline,
    AudioMetricProjector,
    audio_feature_fusion,
    BYOLModel,
    BYOLProjectionHead,
    BYOLPredictionHead,
    byol_loss,
    CatalogExtractor,
    CheckpointMeta,
    DevEvaluator,
    DistillationDataset,
    DistillationProjection,
    distillation_loss,
    ExperimentConfig,
    FMAPackedDataset,
    LateInteractionReranker,
    LeakageGuard,
    ResourceLog,
    SupConLoss,
    V5PairResolver,
)

ROOT = Path(__file__).resolve().parents[1]
V5_BENCH = ROOT / "benchmarks" / "soundalike_pairs.v5.json"


# ═══════════════════════════════════════════════════════════════════════════
# A. ExperimentConfig
# ═══════════════════════════════════════════════════════════════════════════


def test_config_defaults_present():
    cfg = ExperimentConfig()
    # Core protocol fields.
    assert cfg.protocol_version == "v5"
    assert cfg.benchmark_id == "soundalike-pure-sonic-v5"
    # Metric fields match v5 policy.
    assert cfg.recall_cutoffs == (1, 5, 10, 20, 50)
    assert cfg.ndcg_cutoffs == (10, 50)
    assert cfg.candidate_recall_at == (50, 200, 1000)
    assert cfg.bootstrap_seed == 20260711
    # All SupCon / BYOL / distill dims are positive integers.
    assert cfg.supcon_embedding_dim > 0
    assert cfg.byol_embedding_dim > 0
    assert cfg.distill_embedding_dim > 0
    assert cfg.late_interaction_windows == 4


def test_config_json_round_trip():
    cfg = ExperimentConfig(seed=9999, supcon_embedding_dim=128)
    j = cfg.to_json()
    d = json.loads(j)
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.seed == 9999
    assert cfg2.supcon_embedding_dim == 128
    # Tuple fields survive JSON round-trip.
    assert isinstance(cfg2.recall_cutoffs, tuple)
    assert cfg2.recall_cutoffs == cfg.recall_cutoffs


def test_config_hash_stable_across_instances():
    a = ExperimentConfig(seed=1)
    b = ExperimentConfig(seed=1)
    assert a.config_hash() == b.config_hash()


def test_config_hash_changes_with_params():
    a = ExperimentConfig(seed=1)
    b = ExperimentConfig(seed=2)
    assert a.config_hash() != b.config_hash()


def test_config_hash_length():
    assert len(ExperimentConfig().config_hash()) == 12


def test_config_is_frozen():
    cfg = ExperimentConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.seed = 0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# B. CheckpointMeta
# ═══════════════════════════════════════════════════════════════════════════


def test_checkpoint_meta_round_trip(tmp_path):
    meta = CheckpointMeta(
        phase="supcon",
        epoch=5,
        loss=0.42,
        config_hash="abc123",
        timestamp="2026-07-12T00:00:00+00:00",
        wall_seconds=3.14,
        extra={"device": "cpu"},
    )
    out = tmp_path / "meta.json"
    meta.save(out)
    loaded = CheckpointMeta.load(out)
    assert loaded.phase == "supcon"
    assert loaded.epoch == 5
    assert abs(loaded.loss - 0.42) < 1e-9
    assert loaded.extra["device"] == "cpu"


def test_checkpoint_meta_to_dict():
    meta = CheckpointMeta(
        phase="byol", epoch=1, loss=0.1,
        config_hash="xx", timestamp="t", wall_seconds=1.0
    )
    d = meta.to_dict()
    assert d["phase"] == "byol"
    assert "extra" in d


# ═══════════════════════════════════════════════════════════════════════════
# C. ResourceLog
# ═══════════════════════════════════════════════════════════════════════════


def test_resource_log_fields():
    r = ResourceLog(
        phase="extract", wall_seconds=10.0, peak_gpu_mb=0.0,
        peak_ram_mb=0.0, n_rows=1000, rows_per_sec=100.0
    )
    d = r.to_dict()
    assert d["phase"] == "extract"
    assert d["n_rows"] == 1000
    assert d["rows_per_sec"] == pytest.approx(100.0)


# ═══════════════════════════════════════════════════════════════════════════
# D. LeakageGuard
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not V5_BENCH.exists(), reason="v5 benchmark not present")
def test_leakage_guard_loads_v5():
    g = LeakageGuard(V5_BENCH)
    assert g.n_dev == 67
    assert g.n_final == 40
    assert len(g.pairs) == 107


@pytest.mark.skipif(not V5_BENCH.exists(), reason="v5 benchmark not present")
def test_leakage_guard_artist_set_nonempty():
    g = LeakageGuard(V5_BENCH)
    s = g.benchmark_artist_set()
    assert len(s) >= 60  # at least ~1 artist per pair (many pairs share none)
    # All entries are non-empty strings.
    assert all(isinstance(a, str) and len(a) > 0 for a in s)


@pytest.mark.skipif(not V5_BENCH.exists(), reason="v5 benchmark not present")
def test_leakage_guard_exclusion_mask_excludes_benchmark_artist():
    g = LeakageGuard(V5_BENCH)
    # Take the first dev pair's query artist — it must be excluded.
    first_pair = g.dev_pairs()[0]
    query_artist = first_pair["query"]["artist"]
    catalog_artists = [query_artist, "totally unknown artist xyz 999"]
    mask = g.exclusion_mask(catalog_artists)
    assert mask[0]   # benchmark artist excluded
    assert not mask[1]  # unknown artist not excluded


@pytest.mark.skipif(not V5_BENCH.exists(), reason="v5 benchmark not present")
def test_leakage_guard_exclusion_mask_shape():
    g = LeakageGuard(V5_BENCH)
    artists = ["artist_a", "artist_b", "artist_c"]
    mask = g.exclusion_mask(artists)
    assert mask.shape == (3,)
    assert mask.dtype == bool


@pytest.mark.skipif(not V5_BENCH.exists(), reason="v5 benchmark not present")
def test_leakage_guard_dev_pairs_are_dev_split_only():
    g = LeakageGuard(V5_BENCH)
    for pair in g.dev_pairs():
        assert pair["split"] == "development"


@pytest.mark.skipif(not V5_BENCH.exists(), reason="v5 benchmark not present")
def test_leakage_guard_final_pairs_are_final_split_only():
    g = LeakageGuard(V5_BENCH)
    for pair in g.final_pairs():
        assert pair["split"] == "final"


def test_leakage_guard_wrong_id_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "benchmark_id": "wrong-id",
        "schema_version": 5,
        "pairs": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_id"):
        LeakageGuard(bad)


def test_leakage_guard_wrong_schema_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "benchmark_id": "soundalike-pure-sonic-v5",
        "schema_version": 4,
        "pairs": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        LeakageGuard(bad)


# ═══════════════════════════════════════════════════════════════════════════
# E. _stratified_split
# ═══════════════════════════════════════════════════════════════════════════


def test_stratified_split_covers_all_indices():
    genres = np.array(["rock"] * 50 + ["jazz"] * 30 + ["pop"] * 20)
    splits = _stratified_split(genres, val_frac=0.1, test_frac=0.1, seed=0)
    all_idx = np.concatenate([splits["train"], splits["val"], splits["test"]])
    assert len(all_idx) == 100
    assert len(set(all_idx.tolist())) == 100  # disjoint


def test_stratified_split_all_genres_in_each_split():
    genres = np.array(["rock"] * 50 + ["jazz"] * 30 + ["pop"] * 20)
    splits = _stratified_split(genres, val_frac=0.2, test_frac=0.2, seed=1)
    for name in ("train", "val", "test"):
        present = set(genres[splits[name]].tolist())
        assert present == {"rock", "jazz", "pop"}


def test_stratified_split_unlabeled_train_only():
    genres = np.array(["rock"] * 30 + ["unlabeled"] * 20)
    splits = _stratified_split(genres, val_frac=0.2, test_frac=0.2, seed=0)
    assert "unlabeled" not in set(genres[splits["val"]].tolist())
    assert "unlabeled" not in set(genres[splits["test"]].tolist())
    assert (genres[splits["train"]] == "unlabeled").sum() == 20


# ═══════════════════════════════════════════════════════════════════════════
# F. FMAPackedDataset  (uses real fma_packed.npz if present, else synthetic)
# ═══════════════════════════════════════════════════════════════════════════


def _make_synthetic_packed(path: Path, n: int = 60, n_mels: int = 128,
                            frames: int = 512) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, n_mels, frames)).astype(np.float16)
    genres_list = ["rock", "jazz", "pop", "unlabeled"]
    genres = np.array([genres_list[i % 4] for i in range(n)])
    artists = np.array([f"artist_{i % 10}" for i in range(n)])
    track_ids = np.arange(n, dtype=np.int64)
    np.savez(path, X=X, genres=genres, artists=artists, track_ids=track_ids,
             titles=np.array([f"title_{i}" for i in range(n)]))


def test_fma_dataset_shapes(tmp_path):
    packed = tmp_path / "fma.npz"
    _make_synthetic_packed(packed, n=60)
    ds = FMAPackedDataset(packed, crop_frames=128, seed=0, split="train",
                          val_frac=0.1, test_frac=0.1)
    assert len(ds) > 0
    v1, v2, label, artist = ds[0]
    assert v1.shape == (1, 128, 128)
    assert v2.shape == (1, 128, 128)
    assert v1.dtype == torch.float32


def test_fma_dataset_two_views_differ(tmp_path):
    packed = tmp_path / "fma.npz"
    _make_synthetic_packed(packed, n=60)
    ds = FMAPackedDataset(packed, crop_frames=128, seed=7, split="train")
    v1, v2, _, _ = ds[0]
    # Two independent augmentations should differ.
    assert not torch.allclose(v1, v2)


def test_fma_dataset_genre_labels_integer(tmp_path):
    packed = tmp_path / "fma.npz"
    _make_synthetic_packed(packed, n=60)
    ds = FMAPackedDataset(packed, crop_frames=128, seed=0, split="train")
    _, _, label, _ = ds[0]
    assert isinstance(label, int)


def test_fma_dataset_n_classes(tmp_path):
    packed = tmp_path / "fma.npz"
    _make_synthetic_packed(packed, n=60)
    ds = FMAPackedDataset(packed, crop_frames=128, seed=0, split="train")
    # We have rock, jazz, pop (unlabeled excluded from class list).
    assert ds.n_classes == 3


def test_fma_dataset_split_sizes_sum_to_total(tmp_path):
    packed = tmp_path / "fma.npz"
    _make_synthetic_packed(packed, n=90)
    train = FMAPackedDataset(packed, crop_frames=64, seed=0, split="train")
    val = FMAPackedDataset(packed, crop_frames=64, seed=0, split="val")
    test = FMAPackedDataset(packed, crop_frames=64, seed=0, split="test")
    assert len(train) + len(val) + len(test) == 90


def test_fma_dataset_guard_excludes_rows(tmp_path):
    packed = tmp_path / "fma.npz"
    # 60 rows; first 6 have artist "artist_0" (indices 0,10,20,30,40,50).
    _make_synthetic_packed(packed, n=60)

    # Build a minimal synthetic guard that excludes "artist_0".
    bench_path = tmp_path / "bench.json"
    bench_path.write_text(json.dumps({
        "benchmark_id": "soundalike-pure-sonic-v5",
        "schema_version": 5,
        "pairs": [
            {
                "id": "DEV-001", "split": "development", "scene": "pop",
                "query": {"title": "q", "artist": "artist_0"},
                "target": {"title": "t", "artist": "artist_99"},
                "evidence_mode": "sourced", "claim_status": "confirmed",
                "sources": [], "evidence_category": "category_a_sonic",
                "deciding_primary": True, "category_reason": "test",
                "legacy_pair_id": None, "evidence_subtype": "direct",
                "catalog_tier": "popular",
            }
        ],
    }), encoding="utf-8")
    guard = LeakageGuard(bench_path)

    ds_no_guard = FMAPackedDataset(packed, seed=0, split="train", val_frac=0.1,
                                   test_frac=0.1)
    ds_with_guard = FMAPackedDataset(packed, seed=0, split="train", val_frac=0.1,
                                     test_frac=0.1, guard=guard)
    assert len(ds_with_guard) < len(ds_no_guard)


# ═══════════════════════════════════════════════════════════════════════════
# G. SupConLoss
# ═══════════════════════════════════════════════════════════════════════════


def _norm(t: "torch.Tensor") -> "torch.Tensor":
    return torch.nn.functional.normalize(t, dim=1)


def test_supcon_same_label_lower_than_random():
    torch.manual_seed(42)
    criterion = SupConLoss(temperature=0.1, cross_artist=False)

    # Case A: two tight clusters, same label → strong positive signal.
    z_a = _norm(torch.randn(8, 32) * 0.01 + torch.tensor([[1.0, 0] + [0] * 30]))
    z_b = _norm(torch.randn(8, 32) * 0.01 + torch.tensor([[-1.0, 0] + [0] * 30]))
    z = torch.cat([z_a, z_b], dim=0)
    labels_clustered = [0] * 8 + [1] * 8
    loss_clustered = criterion(z, labels_clustered)

    # Case B: random embeddings, random labels → little positive signal.
    z_rand = _norm(torch.randn(16, 32))
    labels_rand = [i % 4 for i in range(16)]
    loss_rand = criterion(z_rand, labels_rand)

    assert float(loss_clustered) < float(loss_rand)
    assert torch.isfinite(loss_clustered)


def test_supcon_cross_artist_removes_same_artist_positives():
    torch.manual_seed(0)
    # 8 samples: label 0, all from "artist_A" (same-artist pairs only).
    z = _norm(torch.randn(8, 32))
    labels = [0] * 8
    artists = ["artist_A"] * 8

    criterion_ca = SupConLoss(temperature=0.1, cross_artist=True)
    # With cross_artist=True, same-artist positives are removed → no valid positives.
    loss_ca = criterion_ca(z, labels, artists)
    assert float(loss_ca) == pytest.approx(0.0, abs=1e-6)

    # Without cross_artist, the same-genre pairs are valid positives.
    criterion_noca = SupConLoss(temperature=0.1, cross_artist=False)
    loss_noca = criterion_noca(z, labels)
    # Loss should be non-trivially positive for random embeddings.
    assert float(loss_noca) > 0


def test_supcon_no_valid_positives_returns_zero():
    """All-different labels → no positives → should return zero (skip batch)."""
    z = _norm(torch.randn(4, 16))
    labels = [0, 1, 2, 3]  # each unique label, no positives
    criterion = SupConLoss(temperature=0.1, cross_artist=False)
    loss = criterion(z, labels)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_supcon_unknown_genres_never_form_positive_pairs():
    raw = torch.randn(4, 8, requires_grad=True)
    features = torch.nn.functional.normalize(raw, dim=1)
    loss = SupConLoss(cross_artist=False)(features, [-1, -1, -1, -1])
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(raw.grad) == 0


def test_supcon_loss_finite_and_non_negative():
    torch.manual_seed(7)
    z = _norm(torch.randn(12, 32))
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2]
    criterion = SupConLoss(temperature=0.2, cross_artist=False)
    loss = criterion(z, labels)
    assert torch.isfinite(loss)
    assert float(loss) >= 0


def test_supcon_gradient_flows():
    """Gradient flows from loss back through the normalised features."""
    # Use a linear layer as a leaf-parameter source so we can check .grad.
    linear = torch.nn.Linear(16, 16, bias=False)
    x = torch.randn(6, 16)
    z = _norm(linear(x))
    criterion = SupConLoss(temperature=0.1, cross_artist=False)
    loss = criterion(z, [0, 0, 1, 1, 2, 2])
    loss.backward()
    assert linear.weight.grad is not None
    assert linear.weight.grad.shape == linear.weight.shape


# ═══════════════════════════════════════════════════════════════════════════
# H. BYOL components
# ═══════════════════════════════════════════════════════════════════════════


def test_byol_projection_head_shape():
    head = BYOLProjectionHead(in_dim=64, hidden_dim=128, out_dim=64)
    x = torch.randn(4, 64)
    out = head(x)
    assert out.shape == (4, 64)


def test_byol_prediction_head_shape():
    head = BYOLPredictionHead(in_dim=64, hidden_dim=32, out_dim=64)
    x = torch.randn(3, 64)
    out = head(x)
    assert out.shape == (3, 64)


def test_byol_model_forward_shapes():
    model = BYOLModel(
        embedding_dim=32, width=8, pool_type="avg",
        projection_dim=32, prediction_dim=16, ema_decay=0.99
    )
    v1 = torch.randn(4, 1, 128, 128)
    v2 = torch.randn(4, 1, 128, 128)
    p1, p2, t1, t2 = model(v1, v2)
    assert p1.shape == (4, 32)
    assert p2.shape == (4, 32)
    assert t1.shape == (4, 32)
    assert t2.shape == (4, 32)


def test_byol_loss_finite_scalar():
    p1 = torch.randn(4, 32)
    p2 = torch.randn(4, 32)
    t1 = torch.randn(4, 32)
    t2 = torch.randn(4, 32)
    loss = byol_loss(p1, p2, t1, t2)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_byol_loss_near_identical_views_lower():
    """Similar views → predictions and targets should align → lower loss."""
    torch.manual_seed(0)
    base = torch.randn(8, 32)
    # Identical views → predictions should match target → loss close to −1.
    loss_good = byol_loss(base, base, base, base)
    # Random unrelated → higher loss.
    loss_bad = byol_loss(
        base, torch.randn(8, 32), torch.randn(8, 32), torch.randn(8, 32)
    )
    assert float(loss_good) < float(loss_bad)


def test_byol_ema_update_moves_toward_online():
    model = BYOLModel(
        embedding_dim=16, width=4, pool_type="avg",
        projection_dim=16, prediction_dim=8, ema_decay=0.0  # full copy
    )
    # Modify online parameters with random noise.
    with torch.no_grad():
        for p in model.online_encoder.parameters():
            p.add_(torch.randn_like(p))

    # Snapshot target before update.
    target_before = [p.clone() for p in model.target_encoder.parameters()]
    online_after = [p.clone() for p in model.online_encoder.parameters()]

    model.update_target(decay=0.0)  # full copy: target ← online

    for t_new, o in zip(model.target_encoder.parameters(), online_after):
        assert torch.allclose(t_new, o, atol=1e-6)


def test_byol_ema_partial_decay():
    model = BYOLModel(
        embedding_dim=16, width=4, pool_type="avg",
        projection_dim=16, prediction_dim=8, ema_decay=0.9
    )
    # Fix online and target to known values.
    with torch.no_grad():
        for p in model.online_encoder.parameters():
            p.fill_(1.0)
        for p in model.target_encoder.parameters():
            p.fill_(0.0)

    model.update_target(decay=0.9)
    # Expected: target = 0.9 * 0.0 + 0.1 * 1.0 = 0.1
    for p in model.target_encoder.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1), atol=1e-5)


def test_byol_target_has_no_grad():
    model = BYOLModel(
        embedding_dim=16, width=4, pool_type="avg",
        projection_dim=16, prediction_dim=8
    )
    for p in model.target_encoder.parameters():
        assert not p.requires_grad
    for p in model.target_projector.parameters():
        assert not p.requires_grad


def test_byol_gradient_does_not_flow_to_target():
    model = BYOLModel(
        embedding_dim=16, width=4, pool_type="avg",
        projection_dim=16, prediction_dim=8
    )
    v1 = torch.randn(2, 1, 128, 64)
    v2 = torch.randn(2, 1, 128, 64)
    p1, p2, t1, t2 = model(v1, v2)
    loss = byol_loss(p1, p2, t1, t2)
    loss.backward()
    # Target parameters must have no gradient (they are not leaf grads).
    for p in model.target_encoder.parameters():
        assert p.grad is None


# ═══════════════════════════════════════════════════════════════════════════
# I. DistillationDataset
# ═══════════════════════════════════════════════════════════════════════════


def _make_guard_excluding_artist_0(tmp_path: Path) -> LeakageGuard:
    bench_path = tmp_path / "bench_distill.json"
    bench_path.write_text(json.dumps({
        "benchmark_id": "soundalike-pure-sonic-v5",
        "schema_version": 5,
        "pairs": [
            {
                "id": "DEV-001", "split": "development", "scene": "pop",
                "query": {"title": "q", "artist": "artist_0"},
                "target": {"title": "t", "artist": "artist_99"},
                "evidence_mode": "sourced", "claim_status": "confirmed",
                "sources": [], "evidence_category": "category_a_sonic",
                "deciding_primary": True, "category_reason": "test",
                "legacy_pair_id": None, "evidence_subtype": "direct",
                "catalog_tier": "popular",
            }
        ],
    }), encoding="utf-8")
    return LeakageGuard(bench_path)


def test_distillation_dataset_excludes_benchmark_artists(tmp_path):
    guard = _make_guard_excluding_artist_0(tmp_path)
    n = 30
    rng = np.random.default_rng(0)
    mels = rng.standard_normal((n, 128, 256)).astype(np.float16)
    # 10 rows are "artist_0" → should be excluded.
    teachers = rng.standard_normal((n, 576)).astype(np.float32)
    artists = ["artist_0" if i < 10 else f"safe_artist_{i}" for i in range(n)]

    ds = DistillationDataset(
        mel_matrix=mels, teacher_matrix=teachers,
        artists=artists, guard=guard, crop_frames=128
    )
    assert len(ds) == 20  # 10 excluded
    assert ds._excluded == 10


def test_distillation_dataset_item_shapes(tmp_path):
    guard = _make_guard_excluding_artist_0(tmp_path)
    n = 20
    rng = np.random.default_rng(1)
    mels = rng.standard_normal((n, 128, 256)).astype(np.float16)
    teachers = rng.standard_normal((n, 576)).astype(np.float32)
    artists = [f"safe_{i}" for i in range(n)]

    ds = DistillationDataset(
        mel_matrix=mels, teacher_matrix=teachers,
        artists=artists, guard=guard, crop_frames=128
    )
    mel_t, teacher_t = ds[0]
    assert mel_t.shape == (1, 128, 128)
    assert teacher_t.shape == (576,)
    assert mel_t.dtype == torch.float32
    assert teacher_t.dtype == torch.float32


def test_distillation_dataset_mismatch_raises(tmp_path):
    guard = _make_guard_excluding_artist_0(tmp_path)
    mels = np.zeros((10, 128, 256), dtype=np.float16)
    teachers = np.zeros((11, 576), dtype=np.float32)
    with pytest.raises(ValueError):
        DistillationDataset(mels, teachers, ["a"] * 10, guard, crop_frames=128)


# ═══════════════════════════════════════════════════════════════════════════
# J. distillation_loss
# ═══════════════════════════════════════════════════════════════════════════


def test_distillation_loss_finite():
    s = torch.randn(8, 32, requires_grad=True)
    t = torch.randn(8, 32)
    loss = distillation_loss(s, t, temperature=0.1)
    assert torch.isfinite(loss)
    assert loss.shape == ()


def test_distillation_loss_gradient_flows():
    s = torch.randn(6, 16, requires_grad=True)
    t = torch.randn(6, 16)
    loss = distillation_loss(s, t)
    loss.backward()
    assert s.grad is not None


def test_distillation_loss_no_gradient_through_teacher():
    s = torch.randn(4, 8, requires_grad=True)
    t = torch.randn(4, 8, requires_grad=True)
    loss = distillation_loss(s, t)
    loss.backward()
    # Teacher is detached → its gradient should be None after backward.
    assert t.grad is None


def test_distillation_loss_perfect_alignment_is_minimum():
    """When student == teacher direction, loss is minimal."""
    v = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    loss_perfect = distillation_loss(v.clone(), v.clone(), temperature=1.0)
    loss_random = distillation_loss(
        torch.nn.functional.normalize(torch.randn(4, 16), dim=1),
        v.clone(), temperature=1.0
    )
    assert float(loss_perfect) < float(loss_random)


# ═══════════════════════════════════════════════════════════════════════════
# K. CatalogExtractor
# ═══════════════════════════════════════════════════════════════════════════


def test_catalog_extractor_output_float16():
    from soundalike.ml.model import ResNetAudioEncoder
    enc = ResNetAudioEncoder(embedding_dim=16, width=4).eval()
    extractor = CatalogExtractor(enc, batch_size=8, device="cpu")
    mels = np.random.randn(20, 128, 256).astype(np.float32)
    emb = extractor.extract(mels, crop_frames=128)
    assert emb.dtype == np.float16
    assert emb.shape == (20, 16)


def test_catalog_extractor_l2_normalized():
    from soundalike.ml.model import ResNetAudioEncoder
    enc = ResNetAudioEncoder(embedding_dim=16, width=4).eval()
    extractor = CatalogExtractor(enc, batch_size=10, device="cpu")
    mels = np.random.randn(15, 128, 256).astype(np.float32)
    emb = extractor.extract(mels, crop_frames=128).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_catalog_extractor_correct_row_count():
    from soundalike.ml.model import ResNetAudioEncoder
    enc = ResNetAudioEncoder(embedding_dim=8, width=4).eval()
    extractor = CatalogExtractor(enc, batch_size=7, device="cpu")
    mels = np.random.randn(25, 128, 200).astype(np.float32)
    emb = extractor.extract(mels, crop_frames=128)
    assert emb.shape[0] == 25


# ═══════════════════════════════════════════════════════════════════════════
# L. V5PairResolver
# ═══════════════════════════════════════════════════════════════════════════


def test_v5_resolver_exact_match():
    titles = ["Bohemian Rhapsody", "Stairway to Heaven", "Hotel California"]
    artists = ["Queen", "Led Zeppelin", "Eagles"]
    resolver = V5PairResolver(titles, artists)

    row = resolver.query_row({"title": "Bohemian Rhapsody", "artist": "Queen"})
    assert row == 0


def test_v5_resolver_normalises_accents():
    titles = ["Cafe Noir"]
    artists = ["Björk"]
    resolver = V5PairResolver(titles, artists)
    # Normalised "bjork" should still match.
    row = resolver.query_row({"title": "Cafe Noir", "artist": "Björk"})
    assert row == 0


def test_v5_resolver_no_match_returns_none():
    titles = ["Song A"]
    artists = ["Artist A"]
    resolver = V5PairResolver(titles, artists)
    row = resolver.query_row({"title": "Song B", "artist": "Artist B"})
    assert row is None


def test_v5_resolver_target_rows_excludes_derivatives():
    titles = ["Song A", "Song A (Live)", "Song A (Remix)"]
    artists = ["Artist A"] * 3
    resolver = V5PairResolver(titles, artists)
    rows = resolver.target_rows({"title": "Song A", "artist": "Artist A"})
    # Only the original (row 0) should be in target_rows (penalty 0).
    assert rows == {0}


def test_v5_resolver_featuring_artist_match():
    titles = ["Collab Track"]
    artists = ["Main Artist featuring Feature Artist"]
    resolver = V5PairResolver(titles, artists)
    row = resolver.query_row({"title": "Collab Track", "artist": "Main Artist"})
    assert row == 0  # primary artist match


# ═══════════════════════════════════════════════════════════════════════════
# M. DevEvaluator — metrics
# ═══════════════════════════════════════════════════════════════════════════


def _make_minimal_v5_bench(tmp_path: Path, pairs: list) -> Path:
    """Write a minimal syntactic v5 benchmark JSON."""
    bench = {
        "benchmark_id": "soundalike-pure-sonic-v5",
        "schema_version": 5,
        "pairs": pairs,
    }
    p = tmp_path / "v5_mini.json"
    p.write_text(json.dumps(bench), encoding="utf-8")
    return p


def _make_pair(i: int, split: str = "development") -> Dict[str, Any]:
    return {
        "id": f"DEV-{i:04d}",
        "split": split,
        "scene": "pop",
        "query": {"title": f"song_q_{i}", "artist": f"qartist_{i}"},
        "target": {"title": f"song_t_{i}", "artist": f"tartist_{i}"},
        "evidence_mode": "sourced",
        "claim_status": "confirmed",
        "sources": [],
        "evidence_category": "category_a_sonic",
        "deciding_primary": True,
        "category_reason": "test",
        "legacy_pair_id": None,
        "evidence_subtype": "direct",
        "catalog_tier": "popular",
    }


def _make_catalog_and_embeddings(
    n: int = 100,
    dim: int = 32,
    seed: int = 0,
) -> tuple:
    rng = np.random.default_rng(seed)
    titles = [f"song_q_{i}" if i < n // 2 else f"song_t_{i - n // 2}"
              for i in range(n)]
    artists = [f"qartist_{i}" if i < n // 2 else f"tartist_{i - n // 2}"
               for i in range(n)]
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return titles, artists, emb.astype(np.float16)


def test_dev_evaluator_recall_at_k_known_rank(tmp_path):
    """Create a catalog where target is exactly at rank 1 → R@1 = 1.0."""
    N = 50
    dim = 16

    pairs = [_make_pair(0)]  # one dev pair
    bench_path = _make_minimal_v5_bench(tmp_path, pairs)
    guard = LeakageGuard(bench_path)

    cfg = ExperimentConfig(
        benchmark_path=str(bench_path),
        recall_cutoffs=(1, 5, 10),
        ndcg_cutoffs=(10,),
        candidate_recall_at=(10,),
        bootstrap_iterations=100,
    )

    # Build catalog: q at row 0, t at row 1, rest random.
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((N, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    # Make query row 0 and target row 1 perfectly similar (target at rank 1).
    emb[1] = emb[0].copy()  # target = query direction → rank 1

    titles = ["song_q_0", "song_t_0"] + [f"other_{i}" for i in range(N - 2)]
    artists = ["qartist_0", "tartist_0"] + [f"rand_{i}" for i in range(N - 2)]

    evaluator = DevEvaluator(guard=guard, titles=titles, artists=artists, cfg=cfg)
    result = evaluator.evaluate(emb.astype(np.float16))

    m = result["metrics"]
    assert m["recall_at_1"] == pytest.approx(1.0)
    assert m["recall_at_5"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx(1.0)
    assert m["ndcg_at_10"] == pytest.approx(1.0 / math.log2(2))  # rank 1


def test_dev_evaluator_target_not_found_gives_zero_rank(tmp_path):
    N = 20
    dim = 8

    pairs = [_make_pair(0)]
    bench_path = _make_minimal_v5_bench(tmp_path, pairs)
    guard = LeakageGuard(bench_path)
    cfg = ExperimentConfig(benchmark_path=str(bench_path))

    # Target title not in catalog → target_found = False → rank = 0.
    titles = ["song_q_0"] + [f"other_{i}" for i in range(N - 1)]
    artists = ["qartist_0"] + [f"rand_{i}" for i in range(N - 1)]
    emb = np.random.randn(N, dim).astype(np.float16)

    evaluator = DevEvaluator(guard=guard, titles=titles, artists=artists, cfg=cfg)
    result = evaluator.evaluate(emb)

    rec = result["pairs"][0]
    assert rec["target_rank"] == 0
    assert not rec["target_found"]


def test_dev_evaluator_mrr_formula(tmp_path):
    """Verify MRR = mean(1/rank) for known ranks."""
    # 3 dev pairs at ranks 1, 2, 4 → MRR = (1 + 0.5 + 0.25) / 3
    N = 30
    dim = 16

    pairs = [_make_pair(i) for i in range(3)]
    bench_path = _make_minimal_v5_bench(tmp_path, pairs)
    guard = LeakageGuard(bench_path)
    cfg = ExperimentConfig(benchmark_path=str(bench_path),
                           recall_cutoffs=(1, 5, 10, 20, 50),
                           ndcg_cutoffs=(10, 50),
                           candidate_recall_at=(50,))

    rng = np.random.default_rng(99)
    emb = rng.standard_normal((N, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    # Pair 0: query=row 0, target=row 15
    # Pair 1: query=row 1, target=row 16
    # Pair 2: query=row 2, target=row 17
    titles = ([f"song_q_{i}" for i in range(3)] +
              [f"other_{i}" for i in range(12)] +
              [f"song_t_{i}" for i in range(3)] +
              [f"rest_{i}" for i in range(N - 18)])
    artists = ([f"qartist_{i}" for i in range(3)] +
               [f"rand_{i}" for i in range(12)] +
               [f"tartist_{i}" for i in range(3)] +
               [f"rest_{i}" for i in range(N - 18)])

    # Force rank for pair 0: make target (row 15) the nearest non-same-artist.
    emb[15] = emb[0] + 1e-4  # almost identical → rank 1
    emb[15] /= np.linalg.norm(emb[15])

    evaluator = DevEvaluator(guard=guard, titles=titles, artists=artists, cfg=cfg)
    result = evaluator.evaluate(emb.astype(np.float16))

    m = result["metrics"]
    assert "mrr" in m
    assert 0.0 <= m["mrr"] <= 1.0


def test_dev_evaluator_ndcg_formula():
    """Verify ndcg_at_k for a single item matches the log2 formula."""
    ndcg_r1 = DevEvaluator.ndcg_at_k(1, 10)
    ndcg_r2 = DevEvaluator.ndcg_at_k(2, 10)
    ndcg_r11 = DevEvaluator.ndcg_at_k(11, 10)
    ndcg_r0 = DevEvaluator.ndcg_at_k(0, 10)  # not found

    assert ndcg_r1 == pytest.approx(1.0 / math.log2(2))
    assert ndcg_r2 == pytest.approx(1.0 / math.log2(3))
    assert ndcg_r1 > ndcg_r2  # higher rank = higher NDCG
    assert ndcg_r11 == 0.0    # outside cutoff
    assert ndcg_r0 == 0.0     # not found


def test_dev_evaluator_primary_metric_formula(tmp_path):
    """primary = mean(ndcg_at_10, mrr, recall_at_10)."""
    N = 20
    dim = 8
    pairs = [_make_pair(0)]
    bench_path = _make_minimal_v5_bench(tmp_path, pairs)
    guard = LeakageGuard(bench_path)
    cfg = ExperimentConfig(benchmark_path=str(bench_path))

    # Target at rank 1 → ndcg=1/log2(2), mrr=1, r@10=1
    emb = np.random.randn(N, dim).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    titles = ["song_q_0", "song_t_0"] + [f"x_{i}" for i in range(N - 2)]
    artists = ["qartist_0", "tartist_0"] + [f"z_{i}" for i in range(N - 2)]
    emb[1] = emb[0]  # target at rank 1

    evaluator = DevEvaluator(guard=guard, titles=titles, artists=artists, cfg=cfg)
    result = evaluator.evaluate(emb.astype(np.float16))
    m = result["metrics"]

    expected_primary = (m["ndcg_at_10"] + m["mrr"] + m["recall_at_10"]) / 3.0
    assert m["primary"] == pytest.approx(expected_primary, rel=1e-6)


def test_dev_evaluator_candidate_recall_at_k(tmp_path):
    """Target at rank 3 → candidate_recall_at_50=1, candidate_recall_at_1=0."""
    N = 50
    dim = 16

    pairs = [_make_pair(0)]
    bench_path = _make_minimal_v5_bench(tmp_path, pairs)
    guard = LeakageGuard(bench_path)
    cfg = ExperimentConfig(
        benchmark_path=str(bench_path),
        candidate_recall_at=(1, 5, 50),
    )

    rng = np.random.default_rng(5)
    emb = rng.standard_normal((N, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    titles = ["song_q_0", "song_t_0"] + [f"other_{i}" for i in range(N - 2)]
    artists = ["qartist_0", "tartist_0"] + [f"rand_{i}" for i in range(N - 2)]

    # Put target at cosine rank 3 (not rank 1).
    q = emb[0].copy()
    emb[1] = q * 0.95 + 0.05 * rng.standard_normal(dim)  # close but not closest
    emb[1] /= np.linalg.norm(emb[1])
    # Make rows 2,3 slightly more similar than target.
    for r in [2, 3]:
        emb[r] = q * 0.999 + 1e-4 * rng.standard_normal(dim)
        emb[r] /= np.linalg.norm(emb[r])
        # Rename to different artists so they're not filtered.

    evaluator = DevEvaluator(guard=guard, titles=titles, artists=artists, cfg=cfg)
    result = evaluator.evaluate(emb.astype(np.float16))
    m = result["metrics"]

    # Target is in the catalog and within top-50 cosine.
    assert m["candidate_recall_at_50"] == pytest.approx(1.0)


def test_dev_evaluator_excludes_final_split(tmp_path):
    """Final-split pairs must never appear in evaluation output."""
    pairs = [_make_pair(i, split="development") for i in range(3)]
    final_pairs = [_make_pair(10 + i, split="final") for i in range(2)]
    bench_path = _make_minimal_v5_bench(tmp_path, pairs + final_pairs)
    guard = LeakageGuard(bench_path)
    cfg = ExperimentConfig(benchmark_path=str(bench_path))

    N = 30
    emb = np.random.randn(N, 8).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    titles = [f"t_{i}" for i in range(N)]
    artists = [f"a_{i}" for i in range(N)]

    evaluator = DevEvaluator(guard=guard, titles=titles, artists=artists, cfg=cfg)
    result = evaluator.evaluate(emb.astype(np.float16))

    # Only dev pairs in output.
    for rec in result["pairs"]:
        assert rec["split"] == "development"
    assert result["n_dev_pairs"] == 3


def test_dev_evaluator_bootstrap_ci_shape(tmp_path):
    N = 20
    dim = 8
    pairs = [_make_pair(i) for i in range(3)]
    bench_path = _make_minimal_v5_bench(tmp_path, pairs)
    guard = LeakageGuard(bench_path)
    cfg = ExperimentConfig(
        benchmark_path=str(bench_path),
        bootstrap_iterations=200,  # fast for unit test
        bootstrap_seed=0,
    )

    emb = np.random.randn(N, dim).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    titles = [f"song_q_{i}" if i < 3 else f"song_t_{i - 3}" if i < 6
              else f"other_{i}" for i in range(N)]
    artists = [f"qartist_{i}" if i < 3 else f"tartist_{i - 3}" if i < 6
               else f"rand_{i}" for i in range(N)]

    evaluator = DevEvaluator(guard=guard, titles=titles, artists=artists, cfg=cfg)
    result = evaluator.evaluate(emb.astype(np.float16), bootstrap=True)

    ci = result["metrics"]["primary_bootstrap_ci95"]
    assert "ci95_low" in ci
    assert "ci95_high" in ci
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]
    assert ci["n_iterations"] == 200


# ═══════════════════════════════════════════════════════════════════════════
# N. LateInteractionReranker
# ═══════════════════════════════════════════════════════════════════════════


def test_maxsim_score_identical_query_candidate():
    """Query == candidate → MaxSim = 1 for each window → mean = 1."""
    n_win, dim = 4, 16
    q = np.random.randn(n_win, dim).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # Single candidate identical to query.
    cand = q[np.newaxis]  # (1, 4, 16)
    scores = LateInteractionReranker.maxsim_score(q, cand)
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_maxsim_score_shape():
    q = np.random.randn(4, 16).astype(np.float32)
    cands = np.random.randn(20, 4, 16).astype(np.float32)
    scores = LateInteractionReranker.maxsim_score(q, cands)
    assert scores.shape == (20,)


def test_maxsim_score_similar_candidate_ranks_higher():
    """A candidate that matches the query should outrank a random one."""
    rng = np.random.default_rng(3)
    n_win, dim = 4, 32
    q = rng.standard_normal((n_win, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    # Similar: q + tiny noise.
    sim = q + 0.01 * rng.standard_normal((n_win, dim)).astype(np.float32)
    sim /= np.linalg.norm(sim, axis=1, keepdims=True)

    # Random: unrelated.
    rand = rng.standard_normal((n_win, dim)).astype(np.float32)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)

    cands = np.stack([sim, rand], axis=0)  # (2, 4, 32)
    scores = LateInteractionReranker.maxsim_score(q, cands)
    assert scores[0] > scores[1]  # similar ranks higher


def test_late_interaction_reranker_extract_windows():
    from soundalike.ml.model import ResNetAudioEncoder

    enc = ResNetAudioEncoder(embedding_dim=16, width=4).eval()
    reranker = LateInteractionReranker(
        encoder=enc, n_windows=4, window_frames=64, batch_size=4, device="cpu"
    )
    mels = np.random.randn(10, 128, 256).astype(np.float32)
    windows = reranker.extract_windows(mels)
    assert windows.shape == (10, 4, 16)
    assert windows.dtype == np.float16


def test_late_interaction_reranker_rerank_order():
    """Verify rerank returns a permutation of candidate_indices."""
    from soundalike.ml.model import ResNetAudioEncoder

    enc = ResNetAudioEncoder(embedding_dim=16, width=4).eval()
    reranker = LateInteractionReranker(
        encoder=enc, n_windows=4, window_frames=64, batch_size=4, device="cpu"
    )
    mels = np.random.randn(20, 128, 256).astype(np.float32)
    all_wins = reranker.extract_windows(mels)

    q_wins = all_wins[0]
    candidates = list(range(1, 10))
    reranked = reranker.rerank(q_wins, candidates, all_wins)

    assert sorted(reranked) == sorted(candidates)  # same elements
    assert len(reranked) == len(candidates)


# ═══════════════════════════════════════════════════════════════════════════
# O. AudioExperimentPipeline — smoke tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not V5_BENCH.exists(), reason="v5 benchmark not present")
def test_pipeline_evaluate_dev_smoke():
    """Pipeline.evaluate_dev runs without error on synthetic embeddings."""
    cfg = ExperimentConfig(
        benchmark_path=str(V5_BENCH),
        candidate_recall_at=(50, 200),
        bootstrap_iterations=0,  # skip bootstrap in smoke test
    )
    pipe = AudioExperimentPipeline(cfg)

    # Build a synthetic catalog large enough to resolve some pairs.
    N = 50
    dim = 32
    emb = np.random.randn(N, dim).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    titles = [f"synth_title_{i}" for i in range(N)]
    artists = [f"synth_artist_{i}" for i in range(N)]

    result = pipe.evaluate_dev(
        embeddings=emb.astype(np.float16),
        titles=titles,
        artists=artists,
        method_name="smoke_test",
    )

    assert result["method"] == "smoke_test"
    assert "metrics" in result
    assert "pairs" in result
    assert "config_hash" in result
    # No final-split pairs leak into results.
    for pair in result["pairs"]:
        assert pair["split"] == "development"


@pytest.mark.skipif(not V5_BENCH.exists(), reason="v5 benchmark not present")
def test_pipeline_evaluate_dev_returns_all_metric_keys():
    cfg = ExperimentConfig(benchmark_path=str(V5_BENCH))
    pipe = AudioExperimentPipeline(cfg)

    N = 30
    emb = np.random.randn(N, 16).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    result = pipe.evaluate_dev(
        embeddings=emb.astype(np.float16),
        titles=[f"t_{i}" for i in range(N)],
        artists=[f"a_{i}" for i in range(N)],
    )
    m = result["metrics"]
    for k in (1, 5, 10, 20, 50):
        assert f"recall_at_{k}" in m
    assert "mrr" in m
    assert "ndcg_at_10" in m
    assert "ndcg_at_50" in m
    assert "primary" in m
    assert "candidate_recall_at_50" in m
    assert "candidate_recall_at_200" in m
    assert "candidate_recall_at_1000" in m


def test_pipeline_config_seed_reproducibility(tmp_path):
    """Same seed → same stratified split indices."""
    packed = tmp_path / "fma.npz"
    _make_synthetic_packed(packed, n=100)

    cfg1 = ExperimentConfig(
        fma_packed_path=str(packed),
        benchmark_path=str(V5_BENCH) if V5_BENCH.exists() else "nonexistent",
        train_seed=123,
    )
    # Use _stratified_split directly (no benchmark needed).
    genres = np.array(["rock"] * 40 + ["jazz"] * 30 + ["pop"] * 30)
    sp1 = _stratified_split(genres, val_frac=0.1, test_frac=0.1, seed=cfg1.train_seed)
    sp2 = _stratified_split(genres, val_frac=0.1, test_frac=0.1, seed=cfg1.train_seed)
    np.testing.assert_array_equal(sp1["train"], sp2["train"])
    np.testing.assert_array_equal(sp1["val"], sp2["val"])


def test_pipeline_resource_log_populated(tmp_path):
    packed = tmp_path / "fma.npz"
    _make_synthetic_packed(packed, n=40)
    if not V5_BENCH.exists():
        pytest.skip("v5 benchmark not present")

    cfg = ExperimentConfig(
        benchmark_path=str(V5_BENCH),
        fma_packed_path=str(packed),
    )
    pipe = AudioExperimentPipeline(cfg)
    from soundalike.ml.model import ResNetAudioEncoder
    enc = ResNetAudioEncoder(embedding_dim=8, width=4)
    mels = np.random.randn(20, 128, 256).astype(np.float32)
    _ = pipe.extract_catalog(enc, mels, crop_frames=128)

    summary = pipe.resource_summary()
    assert len(summary) == 1
    assert summary[0]["phase"] == "extract"
    assert summary[0]["n_rows"] == 20
    assert summary[0]["wall_seconds"] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# P. _build_teacher utility
# ═══════════════════════════════════════════════════════════════════════════


def test_build_teacher_concatenates_and_normalises():
    rng = np.random.default_rng(0)
    sonic = rng.standard_normal((20, 64)).astype(np.float32)
    clap = rng.standard_normal((20, 512)).astype(np.float32)
    teacher = _build_teacher(sonic, clap)
    assert teacher.shape == (20, 576)
    norms = np.linalg.norm(teacher, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_teacher_dtype():
    sonic = np.ones((5, 4), dtype=np.float16)
    clap = np.ones((5, 8), dtype=np.float16)
    teacher = _build_teacher(sonic, clap)
    assert teacher.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════
# Q. Text normalisation helpers
# ═══════════════════════════════════════════════════════════════════════════


def test_normalize_removes_accents():
    assert _normalize("Björk") == "bjork"


def test_normalize_brackets():
    assert _normalize("Song (2019 Remaster)") == "song"


def test_primary_artist_feat():
    assert _primary_artist("Drake featuring Lil Wayne") == "drake"


def test_primary_artist_ft():
    assert _primary_artist("Artist ft. Other") == "artist"


def test_credited_artists_splits_collaborations():
    parts = _credited_artists("Artist A & Artist B")
    assert "artist a" in parts
    assert "artist b" in parts


def test_audio_feature_fusion_group_normalises_rows():
    first = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    second = np.asarray([[0.0, 2.0], [0.0, 5.0]], dtype=np.float32)
    fused = audio_feature_fusion(first, second)
    assert fused.shape == (2, 4)
    assert np.allclose(np.linalg.norm(fused, axis=1), 1.0, atol=1e-6)


def test_audio_feature_fusion_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="row counts"):
        audio_feature_fusion(np.ones((2, 3)), np.ones((3, 3)))


def test_audio_metric_projector_outputs_unit_vectors():
    model = AudioMetricProjector(input_dim=6, hidden_dim=8, output_dim=4)
    output = model(torch.randn(5, 6))
    assert output.shape == (5, 4)
    assert torch.allclose(output.norm(dim=1), torch.ones(5), atol=1e-5)
