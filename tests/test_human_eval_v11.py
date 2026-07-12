"""Audio readiness, privacy, and parity tests for the blinded evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import http.client
import json
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from soundalike.ml.human_aggregate_v10 import aggregate, sign_export
from soundalike.ml.human_eval_v10 import approve_export, content_hash
from soundalike.ml.human_eval_v11 import (
    ERRATUM_IDENTITY,
    ERRATUM_NAMESPACE,
    TRUSTED_ERRATUM_ALLOWED_SIGNERS_FILE,
    TRUSTED_V11_ERRATUM,
    audit_preview_coverage,
    evaluator_handler,
    ranking_order_hash,
)

ROOT = Path(__file__).parents[1]
OLD = (
    ROOT / ".goals" / "human-quality-recommendations"
    / "protocol-v10-human-development"
)
NEW = (
    ROOT / ".goals" / "human-quality-recommendations"
    / "protocol-v11-audio-access-erratum"
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_prior_signed_protocol_is_byte_for_byte_unchanged():
    expected = {
        "protocol-v10.json": "d5b16b6268bc8675a97b66f945531e0b648beeaf637d9e931156ffdd60de059c",
        "served-lists-v10.json": "05e50613e5c5e2e9633ebbf67adc318f45651b2ca93df8ab0c11e12a12b38f8b",
        "state.json": "f0e942b2f3585668d611ec59d397cb834c81400aa19f04ae0377c4c03eb70350",
        "state.sig": "e43f1c5286ce037526674a6b25e25260f39269a1dc50fb6763f8e00e77a1a0ad",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((OLD / name).read_bytes()).hexdigest() == digest


def test_audio_pack_has_full_stable_id_coverage_and_exact_order_parity():
    old, new = _json(OLD / "served-lists-v10.json"), _json(NEW / "served-lists-v11.json")
    protocol, erratum = _json(NEW / "protocol-v11.json"), _json(
        NEW / "audio-access-erratum-v11.json"
    )
    assert content_hash(new) == new["content_sha256"] == protocol["served_lists_sha256"]
    assert ranking_order_hash(old) == ranking_order_hash(new)
    assert erratum["old_list_order_sha256"] == erratum["new_list_order_sha256"]
    assert erratum["old_list_order_sha256"] == ranking_order_hash(new)
    assert erratum["list_order_semantically_identical"] is True
    assert protocol["private_key_sha256"] == erratum["private_method_key_sha256"]
    results = [row for seed in new["seeds"] for row in seed["results"]]
    assert len(results) == len({row["result_id"] for row in results}) == 480
    assert all(row["deezer_track_id"] == row["track_id"] for row in results)
    assert all(
        seed["query"]["deezer_track_id"] == seed["query"]["track_id"]
        for seed in new["seeds"]
    )
    assert all(
        "preview_url" not in row and "preview_available" not in row
        for row in results + [seed["query"] for seed in new["seeds"]]
    )
    assert "cdnt-preview" not in (NEW / "served-lists-v11.json").read_text(
        encoding="utf-8"
    )


def test_audio_erratum_signature_and_blinding():
    executable = shutil.which("ssh-keygen")
    if executable is None:
        pytest.skip("ssh-keygen is unavailable")
    artifact = NEW / "audio-access-erratum-v11.json"
    assert _json(artifact)["content_sha256"] == TRUSTED_V11_ERRATUM
    assert hashlib.sha256(
        (NEW / "erratum-allowed-signers").read_bytes()
    ).hexdigest() == TRUSTED_ERRATUM_ALLOWED_SIGNERS_FILE
    verified = subprocess.run(
        [
            executable, "-Y", "verify",
            "-f", str(NEW / "erratum-allowed-signers"),
            "-I", ERRATUM_IDENTITY,
            "-n", ERRATUM_NAMESPACE,
            "-s", str(NEW / "audio-access-erratum-v11.sig"),
        ],
        input=artifact.read_bytes(),
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr.decode(errors="replace")
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            NEW / "protocol-v11.json",
            NEW / "served-lists-v11.json",
            NEW / "audio-access-erratum-v11.json",
        )
    )
    for secret in ('"method_role"', '"production_baseline"', '"challenger"'):
        assert secret not in public_text


def test_evaluator_network_privacy_and_local_only_state():
    html = (ROOT / "benchmarks" / "human_eval_v11.html").read_text(encoding="utf-8")
    assert "connect-src 'self' https://soundalike.yassin.app" in html
    assert "media-src https://*.dzcdn.net" in html
    assert '<meta name="referrer" content="no-referrer">' in html
    assert 'credentials:"omit",cache:"no-store",referrerPolicy:"no-referrer"' in html
    assert 'url.searchParams.set("id",String(deezerId))' in html
    assert 'const previewCache=new Map()' in html
    assert "previewCache" not in html[html.index("function save()"):html.index("async function hmac")]
    assert "restoreAutosave" in html and "localStorage.getItem(pointerKey())" in html
    assert "rating values" in html and "session ID" in html
    assert 'rel="noopener noreferrer" referrerpolicy="no-referrer"' in html
    assert "https://www.deezer.com/track/" in html
    assert "https://open.spotify.com/search/" in html
    assert "NO_PREVIEW" in html and "skip the sonic rating" in html
    assert "production_baseline" not in html and "challenger" not in html
    assert "<script src=" not in html and "No analytics" in html
    assert "integrity_hmac_sha256" in html and "localStorage" in html


def test_loopback_server_rejects_extra_parameters_and_does_not_cache(
    monkeypatch, tmp_path
):
    import soundalike.ml.human_eval_v11 as module

    monkeypatch.setattr(
        module, "_fresh_deezer_preview",
        lambda track_id: f"https://cdnt-preview.dzcdn.net/{track_id}.mp3",
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        evaluator_handler(
            ROOT / "benchmarks" / "human_eval_v11.html",
            NEW / "protocol-v11.json",
            NEW / "served-lists-v11.json",
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base}/api/preview?id=123", timeout=5) as response:
            body = json.loads(response.read())
            assert body == {
                "ok": True,
                "preview": "https://cdnt-preview.dzcdn.net/123.mp3",
            }
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["Referrer-Policy"] == "no-referrer"
        for suffix in ("?id=123&rating=9", "?id=123&id=456", "?id=abc", "?id=0"):
            with pytest.raises(HTTPError) as error:
                urlopen(f"{base}/api/preview{suffix}", timeout=5)
            assert error.value.code == 400
        with urlopen(f"{base}/protocol.json", timeout=5) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert response.headers["Content-Security-Policy"] == (
                "frame-ancestors 'none'; base-uri 'none'"
            )
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        connection.putrequest("GET", "/", skip_host=True)
        connection.putheader("Host", "attacker.example")
        connection.endheaders()
        assert connection.getresponse().status == 421
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_production_preview_handler_accepts_only_one_numeric_id():
    path = ROOT / "webapp" / "api" / "preview.py"
    spec = importlib.util.spec_from_file_location("preview_api_v11_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._track_id("/api/preview?id=71645431") == "71645431"
    for value in (
        "/api/preview",
        "/api/preview?id=x",
        "/api/preview?id=0",
        "/api/preview?id=1&rating=9",
        "/api/preview?id=1&id=2",
        "/api/preview?id=123456789012345678901",
    ):
        assert module._track_id(value) is None
    source = path.read_text(encoding="utf-8")
    assert '"Cache-Control", "no-store"' in source
    assert '"Referrer-Policy", "no-referrer"' in source
    assert '"cover"' not in source


def test_preview_audit_counts_positions_without_persisting_urls(monkeypatch):
    import soundalike.ml.human_eval_v11 as module

    monkeypatch.setattr(
        module, "_probe",
        lambda endpoint, track_id: {
            "status": "no_preview" if track_id % 17 == 0 else "available",
            "http_status": 404 if track_id % 17 == 0 else 200,
        },
    )
    report = audit_preview_coverage(
        NEW / "served-lists-v11.json",
        "https://soundalike.yassin.app/api/preview",
    )
    assert report["unique_results"]["total"] == report["unique_results"]["id_covered"] == 480
    assert report["seeds"]["total"] == report["seeds"]["id_covered"] == 60
    assert report["ranked_positions"]["total"] == 600
    assert report["external_request"]["ratings_rater_session_transmitted"] is False
    assert report["external_request"]["signed_preview_urls_persisted"] is False
    assert "dzcdn.net" not in json.dumps(report)


def test_committed_live_audit_meets_audio_target_and_records_chrome():
    report = _json(
        ROOT / ".goals" / "human-quality-recommendations"
        / "artifacts" / "human-eval-preview-audit-v11.json"
    )
    assert content_hash(report) == report["content_sha256"]
    assert report["unique_results"] == {
        "total": 480, "id_covered": 480, "available": 457,
        "no_preview": 23, "errors": 0,
    }
    assert report["seeds"] == {
        "total": 60, "id_covered": 60, "available": 59,
        "no_preview": 1, "errors": 0,
    }
    assert report["ranked_positions"]["available"] == 558
    assert report["ranked_positions"]["resolvable_fraction"] == pytest.approx(0.93)
    assert report["chrome_playback"]["status"] == "verified"
    assert report["chrome_playback"]["available_sample"]["playback_completed"] is True
    assert report["chrome_playback"]["privacy_observation"][
        "rating_requests_observed"
    ] == 0
    assert report["external_request"]["signed_preview_urls_persisted"] is False


def test_new_pack_remains_aggregator_compatible_when_private_key_is_present(
    tmp_path,
):
    key = ROOT / "ml_data" / "human_eval_v10" / "method-key-v10.json"
    collector = ROOT / "ml_data" / "human_eval_v10" / "collector_signer"
    if not key.is_file() or not collector.is_file():
        pytest.skip("local private study keys are intentionally not committed")
    protocol, lists = _json(NEW / "protocol-v11.json"), _json(
        NEW / "served-lists-v11.json"
    )
    started = datetime.now(timezone.utc) - timedelta(minutes=1)
    seed = lists["seeds"][0]
    result, served_list = seed["results"][0], seed["lists"][0]
    rated = (started + timedelta(seconds=10)).isoformat()
    document = {
        "schema_version": 10,
        "source_kind": "human_listener",
        "provider": "standalone_local_evaluator",
        "anonymous_rater_id": "anon-v11-compatibility-test",
        "session_id": "session-v11-compatibility",
        "protocol_sha256": protocol["content_sha256"],
        "served_lists_sha256": lists["content_sha256"],
        "local_session_key": "abcdef0123456789" * 4,
        "started_at": started.isoformat(),
        "exported_at": (started + timedelta(seconds=30)).isoformat(),
        "duration_ms": 30_000,
        "result_ratings": {
            result["result_id"]: {
                "similarity": "somewhat_similar",
                "score_0_10": 5,
                "junk_or_version": False,
                "rated_at": rated,
                "interaction_ms": 1000,
            }
        },
        "list_ratings": {
            served_list["list_id"]: {
                "whole_list_coherence": "somewhat_coherent",
                "unrelated_positions_1_to_3": 1,
                "rated_at": rated,
                "interaction_ms": 1000,
            }
        },
    }
    document["integrity_hmac_sha256"] = sign_export(
        document, document["local_session_key"]
    )
    export = tmp_path / "v11-compatibility.json"
    export.write_text(json.dumps(document) + "\n", encoding="utf-8")
    approve_export(export, collector)
    report = aggregate(
        NEW / "protocol-v11.json",
        NEW / "served-lists-v11.json",
        key,
        [export],
    )
    assert report["valid_export_count"] == 1
    assert report["served_lists_sha256"] == lists["content_sha256"]
