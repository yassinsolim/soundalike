import hashlib
import json

import numpy as np
import pytest

from soundalike.ml.catalog_direct_v8 import (
    DirectListError,
    LOCKED_SEEDS,
    _parser,
    run_direct_lists,
    validate_judgments,
    write_locked_seed_manifest,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_run(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {"audio_weight": 0.35, "style_weight": 0.25, "style_guard_min": 0.2}
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "seeds.json"
    write_locked_seed_manifest(manifest_path, _sha(policy_path))
    for name in ("index.npz", "graph.npz", "style.npz"):
        (tmp_path / name).write_bytes(name.encode())

    seed_titles = [seed["title"] for seed in LOCKED_SEEDS]
    seed_artists = [seed["artist"] for seed in LOCKED_SEEDS]

    class Rec:
        titles = np.array(
            seed_titles
            + ["Candidate - Karaoke", "Two", "Three", "Four", "Five"]
            + ["Six", "Seven", "Eight", "Nine", "Ten"]
            + ["Safe Candidate"]
        )
        artists = np.array(
            seed_artists
            + ["Junk Artist", "B", "C", "D", "E"]
            + ["F", "G", "H", "I", "J"]
            + ["Safe Artist"]
        )
        track_ids = np.arange(100, 131)
        last_retrieval_mode = "toy-dual-sonic"

    class Styles:
        scene_names = ("rock", "pop")
        direct_mask = np.array([False] * 31)
        confidence = np.array([0.7] * 31)

        def artist_id(self, artist):
            matches = np.flatnonzero(Rec.artists == artist)
            return int(matches[0]) if len(matches) else None

        def artist_vector(self, _artist):
            return np.array([0.8, 0.2], np.float32)

        def style_overlap(self, _left, _right):
            return 0.5

    class Ranker:
        def __init__(self, rec, _graph, styles, policy, quality):
            self.rec, self.styles, self.policy, self.quality = rec, styles, policy, quality

        def audio_scores(self, _row):
            return np.linspace(0.0, 1.0, len(self.rec.titles))

        def recommend(self, query_row, n=5):
            rows = list(range(20, 25))
            if query_row:
                rows[0] = 30
            return {
                "results": [
                    {
                        "position": position,
                        "row": row,
                        "title": str(self.rec.titles[row]),
                        "artist": str(self.rec.artists[row]),
                        "track_id": int(self.rec.track_ids[row]),
                        "rationale": {
                            "G": 0.4,
                            "A": 0.6,
                            "S": 0.5,
                            "source": "graph",
                            "query_mode": "toy-graph",
                        },
                    }
                    for position, row in enumerate(rows[:n], 1)
                ]
            }

    class Production:
        def __init__(self, _rec, _heldout):
            pass

        def rank(self, _row, method, n=5):
            assert method == "dual_sonic"
            return list(range(25, 25 + n))

    def preview(track_id, _title, _artist):
        return {
            "url": "https://preview/%s" % track_id if track_id % 2 else "",
            "status": "available" if track_id % 2 else "missing",
        }

    report = run_direct_lists(
        manifest_path,
        _sha(manifest_path),
        policy_path,
        tmp_path / "index.npz",
        tmp_path / "graph.npz",
        tmp_path / "style.npz",
        recommender_factory=lambda _path: Rec(),
        graph_factory=lambda _path: object(),
        style_factory=lambda _path: Styles(),
        ranker_factory=Ranker,
        production_factory=Production,
        preview_lookup=preview,
    )
    return report


def test_locked_seed_set_has_exact_distinct_count_and_broad_scenes(tmp_path):
    assert len(LOCKED_SEEDS) == 20
    assert len({(seed["artist"], seed["title"]) for seed in LOCKED_SEEDS}) == 20
    assert len({seed["scene"] for seed in LOCKED_SEEDS}) >= 10
    manifest = write_locked_seed_manifest(tmp_path / "lock.json", "a" * 64)
    assert manifest["seed_count"] == 20
    assert manifest["results_inspected"] is False
    assert manifest["inspection_rules"]["required_seed_passes"] == 16
    assert manifest["content_sha256"]


def test_lists_requires_lock_and_exact_manifest_and_policy_hash(tmp_path):
    with pytest.raises(DirectListError, match="lock-seeds"):
        run_direct_lists(
            tmp_path / "missing.json", "0" * 64, {}, "i", "g", "s"
        )

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"audio_weight": 1, "style_weight": 2}))
    manifest = tmp_path / "manifest.json"
    write_locked_seed_manifest(manifest, _sha(policy))
    with pytest.raises(DirectListError, match="three policy"):
        run_direct_lists(manifest, _sha(manifest), policy, "i", "g", "s")

    policy.write_text(
        json.dumps(
            {
                "audio_weight": 1,
                "style_weight": 2,
                "style_guard_min": 0.3,
                "fourth": 4,
            }
        )
    )
    with pytest.raises(DirectListError, match="policy manifest hash"):
        run_direct_lists(manifest, _sha(manifest), policy, "i", "g", "s")


def test_toy_lists_have_rationales_flags_styles_and_preview_status(tmp_path):
    report = _toy_run(tmp_path)
    assert len(report["records"]) == 20
    assert report["target_blind_disclosure"]["human_judgments_included"] is False
    record = report["records"][0]
    assert record["resolution"]["resolver"] == "PairResolver"
    result = record["lists"]["catalog_policy"][0]
    assert set(("G", "A", "S", "source", "query_mode")) <= set(result["rationale"])
    assert result["rationale"]["A_definition"].startswith("audio-derived")
    assert result["style"]["source"] == "audio_propagated"
    assert set(result["flags"]) == {"junk", "duplicate", "seed_variant", "same_artist"}
    assert result["flags"]["junk"]
    statuses = {
        item["preview_status"]
        for values in record["lists"].values()
        for item in values
    }
    assert statuses == {"available", "missing"}
    assert report["method"]["formula"]
    assert report["provenance"]["assets"]["index"]["sha256"]


def _judgments(report):
    values = []
    for record in report["records"]:
        positions = []
        for list_name, results in record["lists"].items():
            positions.extend(
                {
                    "list": list_name,
                    "position": item["position"],
                    "title": item["title"],
                    "artist": item["artist"],
                    "rationale": "Listened and recorded a human position assessment.",
                    "junk": False,
                }
                for item in results
            )
        values.append({"id": record["id"], "pass": True, "positions": positions})
    return {"lists_sha256": report["content_sha256"], "judgments": values}


def test_judgment_gate_uses_human_bool_and_automatic_junk_failure(tmp_path):
    report = _toy_run(tmp_path)
    judgments = _judgments(report)
    result = validate_judgments(report, judgments)
    assert result["effective_passes"] == 19
    assert result["gate_met"]
    assert result["per_seed"][0]["automatic_failure"]
    assert not result["coherence_inferred"]

    for judgment in judgments["judgments"][1:5]:
        judgment["pass"] = False
    result = validate_judgments(report, judgments)
    assert result["effective_passes"] == 15
    assert not result["gate_met"]

    judgments["judgments"][5]["positions"][0]["artist"] = "not inspected"
    with pytest.raises(DirectListError, match="names"):
        validate_judgments(report, judgments)


def test_cli_exposes_only_requested_operations():
    action = next(item for item in _parser()._actions if item.dest == "command")
    assert set(action.choices) == {"lock-seeds", "lists", "validate"}
