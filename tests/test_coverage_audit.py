"""Tests for deterministic local coverage auditing; no network fixture is needed."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.coverage_audit import build_audit_report, main


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _targets(categories: list) -> dict:
    return {
        "schema_version": 1,
        "category_model": "Categories are curated artist-anchor proxies because the index lacks genres.",
        "categories": categories,
    }


def _category(name: str, anchors: list) -> dict:
    return {"name": name, "description": "Curated artist-anchor proxy.", "anchors": anchors}


def _index(path: Path, artists: list) -> Path:
    return _write_json(path, {"entries": [{"artist": artist} for artist in artists]})


def test_report_is_deterministic_and_contains_input_hashes(tmp_path):
    index = _index(tmp_path / "index.json", ["Drake", "Drake", "SZA"])
    targets = _write_json(tmp_path / "targets.json", _targets([
        _category("rap", [{"artist": "Drake", "minimum_tracks": 2}]),
        _category("pop", [{"artist": "SZA", "minimum_tracks": 3}]),
    ]))

    first = build_audit_report(index, targets)
    second = build_audit_report(index, targets)

    assert first == second
    assert len(first["index"]["sha256"]) == 64
    assert len(first["targets"]["sha256"]) == 64
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_audits_the_deployed_npz_format_directly(tmp_path):
    index = tmp_path / "deepvibe_index.npz"
    np.savez(
        index,
        artists=np.asarray([
            "Childish Gambino",
            "Childish Gambino",
            "SZA",
            "",
        ]),
    )
    targets = _write_json(tmp_path / "targets.json", _targets([
        _category(
            "rap-rnb",
            [{"artist": "Childish Gambino", "minimum_tracks": 3}],
        ),
    ]))

    report = build_audit_report(index, targets)

    assert report["index"]["format"] == "npz"
    assert report["index"]["entries"] == 4
    assert report["index"]["unknown_artist_entries"] == 1
    assert report["artist_presence_and_thinness"][0]["observed"] == 2
    assert report["artist_presence_and_thinness"][0]["status"] == "thin"


def test_aliases_count_under_the_curated_anchor(tmp_path):
    index = _index(tmp_path / "index.json", ["Artist, The", "Artist, The"])
    targets = _write_json(tmp_path / "targets.json", _targets([
        _category("proxy", [{"artist": "The Artist", "aliases": ["Artist, The"], "minimum_tracks": 2}]),
    ]))

    report = build_audit_report(index, targets)

    artist = report["artist_presence_and_thinness"][0]
    assert artist["aliases"] == ["Artist, The"]
    assert artist["observed"] == 2
    assert artist["status"] == "covered"


def test_missing_entries_precede_thin_entries_in_targeted_plan(tmp_path):
    index = _index(tmp_path / "index.json", ["Thin Artist"])
    targets = _write_json(tmp_path / "targets.json", _targets([
        _category("zeta", [{"artist": "Thin Artist", "minimum_tracks": 2}]),
        _category("alpha", [{"artist": "Missing Artist", "minimum_tracks": 3}]),
    ]))

    plan = build_audit_report(index, targets)["targeted_crawl_plan"]

    assert [(item["artist"], item["reason"]) for item in plan] == [
        ("Missing Artist", "missing"), ("Thin Artist", "thin"),
    ]
    assert plan[0]["budget"] == {"artists": 1, "tracks": 3, "api_calls": 5}
    assert plan[1]["budget"] == {"artists": 1, "tracks": 1, "api_calls": 3}


def test_audit_makes_no_network_calls(tmp_path, monkeypatch):
    index = _index(tmp_path / "index.json", ["Only Artist"])
    targets = _write_json(tmp_path / "targets.json", _targets([
        _category("proxy", [{"artist": "Only Artist", "minimum_tracks": 1}]),
    ]))

    def deny_network(*_args, **_kwargs):
        raise AssertionError("coverage audit must remain offline")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    assert build_audit_report(index, targets)["targeted_crawl_plan"] == []


def test_cli_writes_sorted_json_report(tmp_path):
    index = _index(tmp_path / "index.json", ["Artist"])
    targets = _write_json(tmp_path / "targets.json", _targets([
        _category("proxy", [{"artist": "Artist", "minimum_tracks": 1}]),
    ]))
    output = tmp_path / "audit.json"

    assert main(["--index", str(index), "--targets", str(targets), "--output", str(output)]) == 0
    assert list(json.loads(output.read_text(encoding="utf-8"))) == sorted(
        json.loads(output.read_text(encoding="utf-8"))
    )
