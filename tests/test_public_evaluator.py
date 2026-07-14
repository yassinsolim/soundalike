"""Security and integrity tests for the hosted blind v16 evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / ".goals"
    / "human-quality-recommendations"
    / "protocol-v16-hosted-human-development"
)
DEPLOY = ROOT / "webapp" / "evaluate"
EVALUATOR = DEPLOY / "index.html"

V15 = {
    "protocol_content": "4ee45316350ed1b4b49ffa2758d09f3479231d832cab81f8ae5c62985c852140",
    "lists_content": "5218bcec24cfb05776ee76af46c86059a89244f15109aded39fb90f5d279b1d4",
    "state_content": "ac69cced7cea708f9192de03f9d44cfa46867dda88ab4aba597156255ed1694f",
    "protocol_file": "c8e1cb92a0a81866b853087441be331c3f4d07142331be5a9c7d6596f367156b",
    "lists_file": "4c0b5cd2e19fcb8233f75bb7a194a14a841800ab42955d6c6ed3fabc2ccd4a85",
    "state_file": "69e62f3613a052244235fe56c1ca95652550227bb2cdd1f66b239b94129e0cbf",
    "signature_file": "2a09eae2af71d4fb8a82dcc879430ed7442e2f9039d08c13ccbb65ac5410927a",
    "evaluator_file": "7b549700f5eab6555d98ffd677e3559bb1d8fb4da5ae5f7a418a10c085258300",
    "semantic_order": "7ecc7a456d0d243a12e93a973309e9f10c2ec1fae2b9ec58cb731f699597865e",
    "served_payload": "4bb93ecae2b384bb2e1b9ecc3d9dc7cbd21fcec0e04a4e7c5c8e9f0cc86bd155",
    "method_assignment": "ab2688c8ba67796da994a6bc8078e62ba926f84f96bab35a6357c8d179a1092f",
    "blinding_salt": "8e5d29395770eda38808b6352aef7761315093de9bfb5db1b97c70d80f85e8a2",
}
V16_CONTENT = {
    "protocol-v16.json": "c94ce615c68cde595b4e48ac5010297d76bedbed52948b10d315a39286117727",
    "served-lists-v16.json": "809b98ae4314b396ffb33f7349fee72c94e1a80a33d84b1661ab83166a52b9e9",
    "state.json": "e91a9fa4a37ba8bb55c68c4186fe95608df8c63b3134ef14f50e68488139963d",
}
V16_FILES = {
    "allowed_signers": "2ed0c27a387d63f9aa6eb3d56f9217eeef7a85d536ae9238c076032c602c7410",
    "collector_allowed_signers": "af787b33d44f2db435dd620fe1fe97c02473d7ff3cf7f33da1f896e979bd7101",
    "collector_signer.pub": "514d96fd9262e5e5f24ebdf5cc0287573f9e85b2f101cbdf5f48c12b598550f8",
    "protocol-v16.json": "2936b184c86e83f0080f6a7c3956860d57d93f802cc066baebde4444041ffdcd",
    "served-lists-v16.json": "d61aee560308dfaee1faee5f5282960a454a1fe45462680951727557d0992918",
    "signature-metadata.json": "a304e6ff2316a466259fae18274d2ca7960793fe2c5e48b5f23e464a2623f7d9",
    "signer.pub": "c06ed267216d32abead2bbe71ed54a5882ba13e45b34f7ad5617fe04d4d8037c",
    "state.json": "0702d6c0cbac77872a84f891f0ee62bf3cb645a217167f83beecadf112865002",
    "state.sig": "66abd03472c76f5bc2efd486f328bbca682e04a39d20fc232670fdc9e4d52701",
}
EVALUATOR_SHA = "f49a190692d3d8e169a594644d1b646ed6f81da084fc55062e5ab554501bc884"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _content_hash(document: dict) -> str:
    payload = {
        key: value for key, value in document.items() if key != "content_sha256"
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _semantic_order_hash(document: dict) -> str:
    order = []
    for seed in document["seeds"]:
        result_ids = {
            str(row["result_id"]): int(row["deezer_track_id"])
            for row in seed["results"]
        }
        order.append(
            {
                "seed_id": str(seed["seed_id"]),
                "seed_deezer_track_id": int(seed["query"]["deezer_track_id"]),
                "lists": [
                    {
                        "list_id": str(item["list_id"]),
                        "ranking": [
                            {
                                "position": int(row["position"]),
                                "result_id": str(row["result_id"]),
                                "deezer_track_id": result_ids[str(row["result_id"])],
                            }
                            for row in item["ranking"]
                        ],
                    }
                    for item in seed["lists"]
                ],
            }
        )
    return hashlib.sha256(_canonical_bytes(order)).hexdigest()


def _served_payload_hash(document: dict) -> str:
    payload = {
        key: document[key]
        for key in (
            "rankings_state",
            "ratings_count_at_freeze",
            "seed_count",
            "scene_count",
            "results_per_method",
            "same_artist_filtered",
            "shared_results_rated_once",
            "stable_id_field",
            "preview_urls_resolved_at_freeze",
            "audio_access",
            "seeds",
        )
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _preview_module():
    path = ROOT / "webapp" / "api" / "preview.py"
    spec = importlib.util.spec_from_file_location("soundalike_preview_security", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v16_hashes_bind_zero_rating_state_and_evaluator():
    protocol = _load(AUDIT / "protocol-v16.json")
    lists = _load(AUDIT / "served-lists-v16.json")
    state = _load(AUDIT / "state.json")

    for name, document in (
        ("protocol-v16.json", protocol),
        ("served-lists-v16.json", lists),
        ("state.json", state),
    ):
        assert document["schema_version"] == 16
        assert _content_hash(document) == document["content_sha256"]
        assert document["content_sha256"] == V16_CONTENT[name]
        assert document["rankings_state"] == "RANKINGS_LOCKED"

    assert protocol["served_lists_sha256"] == lists["content_sha256"]
    assert state["served_lists_sha256"] == lists["content_sha256"]
    assert state["protocol_sha256"] == protocol["content_sha256"]
    assert protocol["evaluator_sha256"] == state["evaluator_sha256"] == EVALUATOR_SHA
    assert _file_hash(EVALUATOR) == EVALUATOR_SHA
    assert (
        protocol["ratings_count_at_freeze"]
        == lists["ratings_count_at_freeze"]
        == state["ratings_count_at_freeze"]
        == state["human_rater_exports_ingested"]
        == 0
    )
    assert state["sonic_human_report_exists"] is False
    assert protocol["recommendation_behavior_changed"] is False
    assert state["recommendation_behavior_changed"] is False


def test_v15_to_v16_semantic_parity_and_supersession_commitments():
    protocol = _load(AUDIT / "protocol-v16.json")
    lists = _load(AUDIT / "served-lists-v16.json")
    supersession = protocol["supersedes_v15"]

    assert supersession == _load(AUDIT / "state.json")["supersedes_v15"]
    for key in ("protocol", "lists", "state"):
        assert supersession[f"old_{key}_sha256"] == V15[f"{key}_content"]
        assert supersession[f"old_{key}_file_sha256"] == V15[f"{key}_file"]
    assert supersession["old_signature_file_sha256"] == V15["signature_file"]
    assert supersession["old_evaluator_sha256"] == V15["evaluator_file"]
    assert _semantic_order_hash(lists) == V15["semantic_order"]
    assert lists["semantic_order_sha256"] == V15["semantic_order"]
    assert supersession["old_semantic_order_sha256"] == V15["semantic_order"]
    assert supersession["new_semantic_order_sha256"] == V15["semantic_order"]
    assert _served_payload_hash(lists) == V15["served_payload"]
    assert supersession["old_served_payload_sha256"] == V15["served_payload"]
    assert supersession["new_served_payload_sha256"] == V15["served_payload"]
    assert supersession["old_method_assignment_sha256"] == V15["method_assignment"]
    assert supersession["new_method_assignment_sha256"] == V15["method_assignment"]
    assert supersession["old_blinding_salt_sha256"] == V15["blinding_salt"]
    assert supersession["new_blinding_salt_sha256"] == V15["blinding_salt"]
    assert supersession["ratings_discarded"] == supersession["ratings_migrated"] == 0
    for key in (
        "ranking_order_parity",
        "served_payload_semantically_identical",
        "method_assignments_identical",
        "blinding_salt_identical",
        "opaque_identifiers_retained",
        "candidate_pack_semantics_identical",
        "collector_public_trust_identical",
        "hosted_only",
    ):
        assert supersession[key] is True
    assert "hosted-only evaluator delivery" in supersession["reason"]
    assert "recommendation rankings and behavior are unchanged" in supersession["reason"]


def test_v16_files_are_byte_locked_and_deployed_copies_match():
    assert set(V16_FILES) == {path.name for path in AUDIT.iterdir() if path.is_file()}
    assert all(_file_hash(AUDIT / name) == digest for name, digest in V16_FILES.items())
    assert (DEPLOY / "protocol.json").read_bytes() == (
        AUDIT / "protocol-v16.json"
    ).read_bytes()
    assert (DEPLOY / "served-lists.json").read_bytes() == (
        AUDIT / "served-lists-v16.json"
    ).read_bytes()
    metadata = _load(AUDIT / "signature-metadata.json")
    assert metadata["state_sha256"] == V16_FILES["state.json"]
    assert metadata["state_content_sha256"] == V16_CONTENT["state.json"]
    assert metadata["signer_public_key_sha256"] == V16_FILES["signer.pub"]
    assert metadata["allowed_signers_sha256"] == V16_FILES["allowed_signers"]
    assert metadata["signature_sha256"] == V16_FILES["state.sig"]


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="ssh-keygen required")
def test_v16_ed25519_state_signature_verifies():
    result = subprocess.run(
        [
            shutil.which("ssh-keygen"),
            "-Y",
            "verify",
            "-f",
            str(AUDIT / "allowed_signers"),
            "-I",
            "soundalike-human-eval-v16",
            "-n",
            "soundalike-human-eval-v16",
            "-s",
            str(AUDIT / "state.sig"),
        ],
        input=(AUDIT / "state.json").read_bytes(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_public_served_schema_is_strictly_blinded():
    lists = _load(DEPLOY / "served-lists.json")
    assert set(lists) == {
        "schema_version",
        "pack_kind",
        "rankings_state",
        "ratings_count_at_freeze",
        "seed_count",
        "scene_count",
        "results_per_method",
        "same_artist_filtered",
        "shared_results_rated_once",
        "stable_id_field",
        "preview_urls_resolved_at_freeze",
        "audio_access",
        "seeds",
        "semantic_order_sha256",
        "content_sha256",
    }
    assert all(
        set(seed) == {"seed_id", "source_seed_id", "scene", "query", "results", "lists"}
        and set(seed["query"])
        == {"track_id", "deezer_track_id", "title", "artist"}
        and all(
            set(result)
            == {"result_id", "track_id", "deezer_track_id", "title", "artist"}
            for result in seed["results"]
        )
        and all(
            set(item) == {"list_id", "ranking"}
            and all(set(row) == {"position", "result_id"} for row in item["ranking"])
            for item in seed["lists"]
        )
        for seed in lists["seeds"]
    )

    forbidden_keys = {
        "method",
        "method_role",
        "method_identity",
        "role",
        "assignment",
        "challenger",
        "production_baseline",
        "blinding_salt",
        "blinding_salt_sha256",
        "private_key",
        "private_method_key",
        "records",
        "signing_private_key",
        "unblinding_map",
    }

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key.lower() not in forbidden_keys
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(lists)


def test_no_unblinding_or_private_key_material_is_public():
    html = EVALUATOR.read_text(encoding="utf-8")
    forbidden_html = (
        "method_role",
        "production_baseline",
        '"challenger"',
        "blinding_salt",
        "private_method_key",
        "unblinding_map",
        "BEGIN OPENSSH PRIVATE KEY",
        V15["method_assignment"],
    )
    assert not any(marker in html for marker in forbidden_html)
    assert set(path.name for path in DEPLOY.iterdir()) == {
        "index.html",
        "protocol.json",
        "served-lists.json",
    }
    assert "collector_signer" not in {path.name for path in AUDIT.iterdir()}
    for path in (*AUDIT.iterdir(), *DEPLOY.iterdir()):
        if path.is_file():
            assert b"PRIVATE KEY" not in path.read_bytes()


def test_evaluator_auto_load_relative_urls_and_client_only_privacy():
    html = EVALUATOR.read_text(encoding="utf-8")
    assert "schema_version:16" in html
    assert "schema_version!==16" in html
    assert "soundalike-human-v16" in html
    assert "human-ratings-v16-" in html
    assert '["./protocol.json","./served-lists.json"]' in html
    assert 'credentials:"omit"' in html
    assert 'cache:"no-store"' in html
    assert 'referrerPolicy:"no-referrer"' in html
    assert 'document.addEventListener("DOMContentLoaded",loadBundled,{once:true})' in html
    assert 'new URL("/api/preview",previewOrigin())' in html
    assert '"soundalike.yassin.app"].includes(location.hostname))return location.origin' in html
    assert 'type="file"' in html
    assert "Resume/import export" in html
    assert "Ratings stay in this browser" in html
    assert "send it to the project owner yourself" in html
    assert "this page never uploads it" in html
    assert "sendBeacon" not in html
    assert "XMLHttpRequest" not in html
    assert 'method:"POST"' not in html
    assert "fetch(" in html  # bundled JSON and preview GETs only


def test_vercel_routes_and_security_headers_cover_evaluator():
    config = _load(ROOT / "webapp" / "vercel.json")
    assert {item["source"]: item["destination"] for item in config["rewrites"]} == {
        "/evaluate": "/evaluate/index.html",
        "/evaluate/": "/evaluate/index.html",
    }
    routes = {
        item["source"]: {header["key"]: header["value"] for header in item["headers"]}
        for item in config["headers"]
    }
    global_headers = routes["/(.*)"] 
    assert global_headers == {
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "X-Frame-Options": "DENY",
    }
    assert routes["/evaluate"] == routes["/evaluate/(.*)"] 
    evaluator_headers = routes["/evaluate"]
    assert evaluator_headers["Cache-Control"] == "no-store, max-age=0"
    csp = evaluator_headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "base-uri 'none'" in csp
    assert "form-action 'none'" in csp
    assert "frame-src 'none'" in csp
    assert "script-src 'unsafe-inline'" in csp
    assert "style-src 'unsafe-inline'" in csp
    assert (
        "connect-src 'self' https://soundalike.yassin.app" in csp
    )
    assert "media-src https://dzcdn.net https://*.dzcdn.net" in csp


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "https://soundalike.yassin.app.evil.example",
        "null",
        "http://localhost.evil.example:3000",
        "http://localhost:3000/path",
        "http://localhost:3000\nhttps://evil.example",
    ],
)
def test_preview_rejects_evil_origins_without_upstream_call(monkeypatch, origin):
    preview = _preview_module()
    request = object.__new__(preview.handler)
    request.path = "/api/preview?id=3135556"
    request.headers = {"Origin": origin}
    sent = []
    request._send = lambda code, obj: sent.append((code, obj))
    monkeypatch.setattr(
        preview.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("evil origin reached upstream"),
    )
    request.do_GET()
    assert sent == [(403, {"ok": False, "error": "origin not allowed"})]


@pytest.mark.parametrize(
    "origin",
    [
        "https://soundalike.yassin.app",
        "http://localhost:3000",
        "https://localhost:8787",
        "http://127.0.0.1:8000",
        "http://[::1]:5173",
    ],
)
def test_preview_allows_only_production_and_loopback_origins(origin):
    preview = _preview_module()
    assert preview._allowed_cors_origin(origin) == origin
    assert preview._allowed_cors_origin("https://evil.example") is None


@pytest.mark.parametrize(
    "track_id",
    [
        "",
        "0",
        "-1",
        "01",
        "1.5",
        "abc",
        " 123",
        "\uff11\uff12\uff13",
        "9007199254740992",
        "1" * 17,
    ],
)
def test_preview_rejects_malformed_track_ids_without_upstream_call(
    monkeypatch, track_id
):
    preview = _preview_module()
    request = object.__new__(preview.handler)
    request.path = f"/api/preview?id={track_id}"
    request.headers = {}
    sent = []
    request._send = lambda code, obj: sent.append((code, obj))
    monkeypatch.setattr(
        preview.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("malformed id reached upstream"),
    )
    request.do_GET()
    assert sent == [(400, {"ok": False, "error": "bad id"})]


class _Upstream:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_preview_accepts_trusted_dzcdn_and_drops_untrusted_cover(monkeypatch):
    preview = _preview_module()
    request = object.__new__(preview.handler)
    request.path = "/api/preview?id=3135556"
    request.headers = {"Origin": "https://soundalike.yassin.app"}
    sent = []
    request._send = lambda code, obj: sent.append((code, obj))
    monkeypatch.setattr(
        preview.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Upstream(
            {
                "preview": "https://cdns-preview-a.dzcdn.net/stream.mp3",
                "album": {"cover_medium": "https://evil.example/cover.jpg"},
            }
        ),
    )
    request.do_GET()
    assert sent == [
        (
            200,
            {
                "ok": True,
                "preview": "https://cdns-preview-a.dzcdn.net/stream.mp3",
                "cover": "",
            },
        )
    ]


@pytest.mark.parametrize(
    "url",
    [
        "http://cdns-preview-a.dzcdn.net/stream.mp3",
        "https://dzcdn.net.evil.example/stream.mp3",
        "https://evil-dzcdn.net/stream.mp3",
        "https://user@dzcdn.net/stream.mp3",
        "https://dzcdn.net:444/stream.mp3",
        "https://dzcdn.net/stream.mp3\nhttps://evil.example",
    ],
)
def test_preview_rejects_untrusted_upstream_urls(monkeypatch, url):
    preview = _preview_module()
    assert preview._trusted_dzcdn_url(url) is False
    request = object.__new__(preview.handler)
    request.path = "/api/preview?id=3135556"
    request.headers = {}
    sent = []
    request._send = lambda code, obj: sent.append((code, obj))
    monkeypatch.setattr(
        preview.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Upstream({"preview": url, "album": {}}),
    )
    request.do_GET()
    assert sent == [(502, {"ok": False, "error": "untrusted preview"})]


def test_preview_cors_header_is_never_wildcard():
    preview = _preview_module()
    for origin, expected in (
        ("https://soundalike.yassin.app", "https://soundalike.yassin.app"),
        ("https://evil.example", None),
        (None, None),
    ):
        request = object.__new__(preview.handler)
        request.headers = {} if origin is None else {"Origin": origin}
        request.wfile = io.BytesIO()
        headers = []
        request.send_response = lambda code: None
        request.send_header = lambda key, value: headers.append((key, value))
        request.end_headers = lambda: None
        request._send(200, {"ok": True})
        values = dict(headers)
        assert values.get("Access-Control-Allow-Origin") == expected
        assert values.get("Access-Control-Allow-Origin") != "*"
        assert values["Cache-Control"] == "no-store"
