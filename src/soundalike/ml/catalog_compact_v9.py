"""Compact repeated DEVELOPMENT details while retaining every selected list."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .catalog_list_gold_v9 import canonical_bytes, sha256_bytes, write_json


def _compact_prediction(item: Mapping[str, Any]) -> Dict[str, Any]:
    value = copy.deepcopy(dict(item))
    for role in ("baseline", "challenger"):
        if isinstance(value.get(role), dict):
            value[role].pop("result_evidence", None)
    value.pop("lists", None)
    return value


def _compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    value = copy.deepcopy(dict(summary))
    if "per_record" in value:
        value["per_record"] = [
            _compact_prediction(item) for item in value["per_record"]
        ]
    return value


def compact_report(document: Mapping[str, Any]) -> Dict[str, Any]:
    value = copy.deepcopy(dict(document))
    nested = value["nested_5fold"]
    nested["aggregate_outer_predictions"] = _compact_summary(
        nested["aggregate_outer_predictions"]
    )
    scene = value["scene_held_out"]
    scene["aggregate_predictions"] = _compact_summary(
        scene["aggregate_predictions"]
    )
    for fold in scene["folds"]:
        fold["summary"] = _compact_summary(fold["summary"])
    value["selected_full_dev_evaluation"] = _compact_summary(
        value["selected_full_dev_evaluation"]
    )
    value["co_primary"]["graded_ndcg_at_10"] = {
        key: copy.deepcopy(nested["aggregate_outer_predictions"][key])
        for key in (
            "seeds", "baseline", "challenger", "absolute_ndcg_gain",
            "relative_ndcg_gain", "bootstrap", "improved", "worsened",
            "unchanged", "per_scene", "worst_scene_relative_change",
            "challenger_junk_count", "gates", "gate_pass",
        )
    }
    for prediction in value["actual_selected_lists"]:
        for role in ("baseline", "challenger"):
            prediction[role].pop("result_evidence", None)
    value["compaction"] = {
        "repeated_nested_and_scene_lists_removed": True,
        "all_60_selected_baseline_and_challenger_top10_lists_retained": True,
        "blind_lists_and_judgments_are_separate_hash_bound_artifacts": True,
    }
    value.pop("content_sha256", None)
    value["content_sha256"] = sha256_bytes(canonical_bytes(value))
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        default=(
            ".goals/human-quality-recommendations/artifacts/"
            "catalog-powered-sonic-dev-v9.json"
        ),
    )
    args = parser.parse_args(argv)
    path = Path(args.report)
    original = path.stat().st_size
    compacted = compact_report(json.loads(path.read_text(encoding="utf-8")))
    write_json(path, compacted)
    print(json.dumps({
        "before_bytes": original,
        "after_bytes": path.stat().st_size,
        "content_sha256": compacted["content_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
