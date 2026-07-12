"""Audio-access erratum and loopback server for the frozen v10 human study.

This module never generates or changes recommendations.  It adds explicit
Deezer IDs to the already-locked public pack, proves ranking-order parity, and
resolves short-lived legal preview URLs only when a listener requests one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

from .human_eval_v10 import canonical_bytes, content_hash, file_hash

ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_PREVIEW_ENDPOINT = "https://soundalike.yassin.app/api/preview"
DEFAULT_OLD_DIR = (
    ROOT / ".goals" / "human-quality-recommendations"
    / "protocol-v10-human-development"
)
DEFAULT_NEW_DIR = (
    ROOT / ".goals" / "human-quality-recommendations"
    / "protocol-v11-audio-access-erratum"
)
EXPECTED_CATALOG_ROWS = 272_853
ERRATUM_IDENTITY = "soundalike-human-eval-audio-erratum"
ERRATUM_NAMESPACE = "soundalike-human-eval-audio-erratum"

# Immutable trust anchors for the one real study.  The v11 signer is deliberately
# pinned here rather than trusted from the artifact directory it signs.
TRUSTED_V10_FILES = {
    "protocol-v10.json": "d5b16b6268bc8675a97b66f945531e0b648beeaf637d9e931156ffdd60de059c",
    "served-lists-v10.json": "05e50613e5c5e2e9633ebbf67adc318f45651b2ca93df8ab0c11e12a12b38f8b",
    "state.json": "f0e942b2f3585668d611ec59d397cb834c81400aa19f04ae0377c4c03eb70350",
    "state.sig": "e43f1c5286ce037526674a6b25e25260f39269a1dc50fb6763f8e00e77a1a0ad",
    "allowed_signers": "19ea28ec4ea6480309f911e65f86473cd4324ae610b4682f02ab9c089f7f4fea",
}
TRUSTED_V10_PROTOCOL = "2ce53bbf601618a8541d881a124e4cb961f5813707e8aa1d30c1495b952ec06a"
TRUSTED_V10_LISTS = "458abcb809378dbc16506c9d0055477bc26d74585416b38f53863b07341c9815"
TRUSTED_V10_KEY = "470c1a774409cff420886053d5bbdc0fba71c793db3fca8a9bc1d6c11caabe55"
TRUSTED_V11_PROTOCOL = "24ab0485833f45350a974eb2aa9bcd499fe89bd9ce2fb620ab19345cf8a329f0"
TRUSTED_V11_LISTS = "e22979e8f2debf3d1880f070dcd185d538ee30a2f6f819982a6238c9fe36757d"
TRUSTED_V11_ERRATUM = "c40a69357e9c8af4de4649105c7c5886b8bfb50c973cc10f59ae44c9af274e27"
TRUSTED_ERRATUM_ALLOWED_SIGNERS_FILE = (
    "2bf04a1db9ee7f52b40007170a7fd6a6c96e934bead3ac86e4f6dc4031aa7d2c"
)


class AudioAccessError(ValueError):
    """The metadata-only audio-access revision is unsafe or malformed."""


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def ranking_order_hash(document: Mapping[str, Any]) -> str:
    """Hash only seed/list/result order and opaque identities."""
    order = [
        {
            "seed_id": seed["seed_id"],
            "lists": [
                {
                    "list_id": item["list_id"],
                    "ranking": [
                        {
                            "position": row["position"],
                            "result_id": row["result_id"],
                        }
                        for row in item["ranking"]
                    ],
                }
                for item in seed["lists"]
            ],
        }
        for seed in document["seeds"]
    ]
    return hashlib.sha256(canonical_bytes(order)).hexdigest()


def _stable_id(row: Mapping[str, Any], catalog_ids: set[int]) -> int:
    value = row.get("track_id")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AudioAccessError("every seed/result must have a numeric Deezer track_id")
    if value not in catalog_ids:
        raise AudioAccessError(f"Deezer track ID {value} is absent from the catalog index")
    return value


def _sign_erratum(directory: Path, artifact: Path) -> Dict[str, Any]:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise AudioAccessError("ssh-keygen is required; erratum signing fails closed")
    with tempfile.TemporaryDirectory(prefix="soundalike-audio-erratum-") as temp:
        private = Path(temp) / "signer"
        generated = subprocess.run(
            [
                executable, "-q", "-t", "ed25519", "-N", "",
                "-C", "soundalike-human-audio-erratum-v11", "-f", str(private),
            ],
            capture_output=True,
            check=False,
        )
        if generated.returncode:
            raise AudioAccessError("Ed25519 erratum key generation failed")
        public = private.with_suffix(".pub").read_text(encoding="utf-8").strip()
        fields = public.split()
        signer = directory / "erratum-signer.pub"
        allowed = directory / "erratum-allowed-signers"
        signer.write_text(public + "\n", encoding="utf-8")
        allowed.write_text(
            f"{ERRATUM_IDENTITY} {fields[0]} {fields[1]}\n", encoding="utf-8"
        )
        signed = subprocess.run(
            [
                executable, "-Y", "sign", "-f", str(private),
                "-n", ERRATUM_NAMESPACE, str(artifact),
            ],
            capture_output=True,
            check=False,
        )
        generated_signature = Path(str(artifact) + ".sig")
        signature = directory / "audio-access-erratum-v11.sig"
        if signed.returncode or not generated_signature.is_file():
            raise AudioAccessError("detached Ed25519 erratum signing failed")
        os.replace(generated_signature, signature)
    metadata = {
        "algorithm": "Ed25519 detached SSH signature",
        "namespace": ERRATUM_NAMESPACE,
        "identity": ERRATUM_IDENTITY,
        "artifact_sha256": file_hash(artifact),
        "signer_public_key_sha256": file_hash(signer),
        "allowed_signers_sha256": file_hash(allowed),
        "signature_sha256": file_hash(signature),
    }
    _write(directory / "erratum-signature-metadata.json", metadata)
    return metadata


def freeze_audio_access_pack(
    old_protocol_path: Union[Path, str],
    old_lists_path: Union[Path, str],
    catalog_index_path: Union[Path, str],
    out_dir: Union[Path, str],
    evaluator_path: Union[Path, str],
) -> Dict[str, Path]:
    """Create a metadata-only revision while preserving every ranked identity."""
    old_protocol_file, old_lists_file = Path(old_protocol_path), Path(old_lists_path)
    old_protocol = json.loads(old_protocol_file.read_text(encoding="utf-8"))
    old_lists = json.loads(old_lists_file.read_text(encoding="utf-8"))
    for name, document in (("protocol", old_protocol), ("served lists", old_lists)):
        if document.get("schema_version") != 10:
            raise AudioAccessError(f"{name} is not schema v10")
        if content_hash(document) != document.get("content_sha256"):
            raise AudioAccessError(f"{name} content hash mismatch")
        if document.get("rankings_state") != "RANKINGS_LOCKED":
            raise AudioAccessError(f"{name} rankings are not locked")
    if old_protocol.get("served_lists_sha256") != old_lists["content_sha256"]:
        raise AudioAccessError("old protocol/list hash mismatch")

    import numpy as np

    index_path = Path(catalog_index_path)
    with np.load(index_path, allow_pickle=False) as index:
        ids = index["track_ids"]
        if len(ids) != EXPECTED_CATALOG_ROWS:
            raise AudioAccessError(
                f"expected {EXPECTED_CATALOG_ROWS:,} catalog rows, got {len(ids):,}"
            )
        catalog_ids = set(map(int, ids))
    evaluator = Path(evaluator_path)
    if not evaluator.is_file():
        raise AudioAccessError("v11 evaluator is missing")

    revised = deepcopy(old_lists)
    revised.pop("content_sha256", None)
    result_count = 0
    for seed in revised["seeds"]:
        query = seed["query"]
        query["deezer_track_id"] = _stable_id(query, catalog_ids)
        query.pop("preview_url", None)
        query.pop("preview_available", None)
        for result in seed["results"]:
            result["deezer_track_id"] = _stable_id(result, catalog_ids)
            result.pop("preview_url", None)
            result.pop("preview_available", None)
            result_count += 1
    revised["predecessor_served_lists_sha256"] = old_lists["content_sha256"]
    revised["source_catalog_index"] = {
        "row_count": EXPECTED_CATALOG_ROWS,
        "file_sha256": file_hash(index_path),
        "stable_id_field": "track_ids (Deezer track IDs)",
        "seed_id_coverage": f"{len(revised['seeds'])}/{len(revised['seeds'])}",
        "unique_result_id_coverage": f"{result_count}/{result_count}",
    }
    revised["audio_access"] = {
        "provider": "Deezer public 30-second previews",
        "resolution": "fresh on demand through /api/preview?id=<numeric_deezer_id>",
        "frozen_signed_cdn_urls": False,
        "browser_cache_scope": "memory only; current page session",
        "refresh_on_playback_failure": True,
        "external_request_disclosure": (
            "Only the numeric Deezer track ID is requested; no rating value, "
            "anonymous rater ID, session ID, or localStorage data is transmitted."
        ),
        "fallbacks": ["Deezer track page", "Spotify title/artist search"],
    }
    revised["content_sha256"] = content_hash(revised)

    old_order = ranking_order_hash(old_lists)
    new_order = ranking_order_hash(revised)
    if old_order != new_order:
        raise AudioAccessError("metadata revision changed frozen ranking order")

    protocol = deepcopy(old_protocol)
    protocol.pop("content_sha256", None)
    protocol.update({
        "protocol_kind": "blinded_served_list_human_listener_audio_access_erratum",
        "served_lists_sha256": revised["content_sha256"],
        "predecessor_served_lists_sha256": old_lists["content_sha256"],
        "ranking_order_sha256": new_order,
        "audio_access_erratum_file": "audio-access-erratum-v11.json",
        "evaluator_sha256": file_hash(evaluator),
        "estimated_workload": (
            "60 seeds; about 480 unique result ratings plus 120 list judgments; "
            "90-150 minutes per complete rater; target at least 3 raters, ideally 5"
        ),
        "audio_privacy_notice": revised["audio_access"]["external_request_disclosure"],
    })
    protocol["content_sha256"] = content_hash(protocol)

    erratum: Dict[str, Any] = {
        "schema_version": 11,
        "artifact_kind": "signed_audio_access_only_erratum",
        "rankings_state": "RANKINGS_LOCKED",
        "ratings_count_at_erratum": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "old_protocol_sha256": old_protocol["content_sha256"],
        "new_protocol_sha256": protocol["content_sha256"],
        "old_served_lists_sha256": old_lists["content_sha256"],
        "new_served_lists_sha256": revised["content_sha256"],
        "old_list_order_sha256": old_order,
        "new_list_order_sha256": new_order,
        "list_order_semantically_identical": old_order == new_order,
        "private_method_key_sha256": old_protocol["private_key_sha256"],
        "method_assignments": "unchanged; the original private method key is reused",
        "blinding": "unchanged opaque seed/result/list identities; no role added",
        "change_scope": (
            "Stable Deezer IDs and on-demand audio-access metadata only; no ranked "
            "track, position, method assignment, candidate generation, or ranking changed."
        ),
        "source_catalog_index": revised["source_catalog_index"],
        "evaluator_sha256": file_hash(evaluator),
    }
    erratum["content_sha256"] = content_hash(erratum)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise AudioAccessError("output directory must be empty; frozen artifacts are immutable")
    paths = {
        "protocol": output / "protocol-v11.json",
        "lists": output / "served-lists-v11.json",
        "erratum": output / "audio-access-erratum-v11.json",
    }
    _write(paths["protocol"], protocol)
    _write(paths["lists"], revised)
    _write(paths["erratum"], erratum)
    for filename in ("collector_allowed_signers", "collector_signer.pub"):
        source = old_protocol_file.parent / filename
        if not source.is_file():
            raise AudioAccessError(f"missing original collector trust file: {filename}")
        shutil.copyfile(source, output / filename)
    _sign_erratum(output, paths["erratum"])
    return paths


def _strict_track_id(path: str) -> int:
    parsed = urlsplit(path)
    params = parse_qs(parsed.query, keep_blank_values=True)
    if (
        parsed.path != "/api/preview"
        or set(params) != {"id"}
        or len(params["id"]) != 1
        or not params["id"][0].isdigit()
        or len(params["id"][0]) > 20
    ):
        raise AudioAccessError("request must contain only one numeric Deezer id")
    value = int(params["id"][0])
    if value <= 0:
        raise AudioAccessError("Deezer id must be positive")
    return value


def _fresh_deezer_preview(track_id: int) -> str | None:
    request = Request(
        f"https://api.deezer.com/track/{track_id}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    preview = payload.get("preview")
    if not preview:
        return None
    parsed = urlsplit(preview)
    if parsed.scheme != "https" or not (
        parsed.hostname == "dzcdn.net"
        or (parsed.hostname or "").endswith(".dzcdn.net")
    ):
        raise AudioAccessError("Deezer returned an untrusted preview origin")
    return preview


def evaluator_handler(
    evaluator: Path, protocol: Path, lists: Path
) -> type[BaseHTTPRequestHandler]:
    """Build the loopback-only evaluator handler (also convenient for tests)."""
    resources = {
        "/": ("text/html; charset=utf-8", evaluator.read_bytes()),
        "/benchmarks/human_eval_v11.html": (
            "text/html; charset=utf-8", evaluator.read_bytes()
        ),
        "/protocol.json": ("application/json", protocol.read_bytes()),
        "/served-lists.json": ("application/json", lists.read_bytes()),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "frame-ancestors 'none'; base-uri 'none'",
            )
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, value: Mapping[str, Any]) -> None:
            self._send(
                code, "application/json",
                json.dumps(value, separators=(",", ":")).encode("utf-8"),
            )

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            expected_host = f"127.0.0.1:{self.server.server_address[1]}"
            if self.headers.get("Host") != expected_host:
                return self._json(
                    HTTPStatus.MISDIRECTED_REQUEST,
                    {"ok": False, "error": "invalid loopback host"},
                )
            parsed = urlsplit(self.path)
            if parsed.path == "/api/preview":
                try:
                    track_id = _strict_track_id(self.path)
                    preview = _fresh_deezer_preview(track_id)
                    if preview is None:
                        return self._json(
                            HTTPStatus.NOT_FOUND,
                            {"ok": False, "error": "no preview"},
                        )
                    return self._json(HTTPStatus.OK, {"ok": True, "preview": preview})
                except AudioAccessError:
                    return self._json(
                        HTTPStatus.BAD_REQUEST, {"ok": False, "error": "bad id"}
                    )
                except Exception:
                    return self._json(
                        HTTPStatus.BAD_GATEWAY,
                        {"ok": False, "error": "preview provider unavailable"},
                    )
            resource = resources.get(parsed.path)
            if resource is None or parsed.query:
                return self._json(HTTPStatus.NOT_FOUND, {"ok": False})
            self._send(HTTPStatus.OK, resource[0], resource[1])

    return Handler


def serve(
    *,
    port: int = 8000,
    directory: Path = DEFAULT_NEW_DIR,
    evaluator: Path = ROOT / "benchmarks" / "human_eval_v11.html",
    open_browser: bool = True,
) -> None:
    protocol, lists = directory / "protocol-v11.json", directory / "served-lists-v11.json"
    for path in (evaluator, protocol, lists):
        if not path.is_file():
            raise AudioAccessError(f"required evaluator resource is missing: {path}")
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), evaluator_handler(evaluator, protocol, lists)
    )
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Private blinded evaluator: {url}", flush=True)
    print("Preview requests send only a numeric Deezer track ID.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _validated_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    production = value.rstrip("/") == PRODUCTION_PREVIEW_ENDPOINT
    loopback = (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        and parsed.path == "/api/preview"
        and not parsed.query
    )
    if not production and not loopback:
        raise AudioAccessError("audit endpoint must be the production or loopback preview API")
    return value.rstrip("/")


def _probe(endpoint: str, track_id: int) -> Dict[str, Any]:
    request_url = f"{endpoint}?{urlencode({'id': str(track_id)})}"
    last_error = "unknown"
    for attempt in range(3):
        try:
            request = Request(
                request_url,
                headers={"Accept": "application/json", "Origin": "null"},
                method="GET",
            )
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
                headers = dict(response.headers.items())
            preview = payload.get("preview")
            if (
                response.status == 200 and payload.get("ok") is True
                and isinstance(preview, str) and preview.startswith("https://")
            ):
                return {
                    "status": "available",
                    "http_status": 200,
                    "cors": headers.get("Access-Control-Allow-Origin"),
                    "cache_control": headers.get("Cache-Control"),
                }
            last_error = f"unexpected response {response.status}"
        except HTTPError as error:
            try:
                payload = json.loads(error.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {}
            if error.code == 404 and payload.get("error") == "no preview":
                return {
                    "status": "no_preview",
                    "http_status": 404,
                    "cors": error.headers.get("Access-Control-Allow-Origin"),
                    "cache_control": error.headers.get("Cache-Control"),
                }
            last_error = f"HTTP {error.code}"
        except Exception as error:  # transport failures are retried and counted
            last_error = type(error).__name__
        time.sleep(0.3 * (attempt + 1))
    return {"status": "error", "error": last_error}


def audit_preview_coverage(
    lists_path: Union[Path, str],
    endpoint: str,
    *,
    workers: int = 10,
) -> Dict[str, Any]:
    """Resolve every seed/result ID without persisting any signed CDN URL."""
    endpoint = _validated_endpoint(endpoint)
    document = json.loads(Path(lists_path).read_text(encoding="utf-8"))
    if content_hash(document) != document.get("content_sha256"):
        raise AudioAccessError("served-list content hash mismatch")
    results = [
        (seed["seed_id"], result)
        for seed in document["seeds"] for result in seed["results"]
    ]
    seeds = [(seed["seed_id"], seed["query"]) for seed in document["seeds"]]
    all_rows = [row for _, row in results] + [row for _, row in seeds]
    if any(
        not isinstance(row.get("deezer_track_id"), int)
        or isinstance(row.get("deezer_track_id"), bool)
        or row["deezer_track_id"] <= 0
        for row in all_rows
    ):
        raise AudioAccessError("stable Deezer ID coverage is incomplete")
    ids = sorted({row["deezer_track_id"] for row in all_rows})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = dict(zip(ids, executor.map(lambda value: _probe(endpoint, value), ids)))

    def count(rows: list[tuple[str, Mapping[str, Any]]]) -> Dict[str, int]:
        statuses = [outcomes[row["deezer_track_id"]]["status"] for _, row in rows]
        return {
            "total": len(rows),
            "id_covered": len(rows),
            "available": statuses.count("available"),
            "no_preview": statuses.count("no_preview"),
            "errors": statuses.count("error"),
        }

    result_status = {
        row["result_id"]: outcomes[row["deezer_track_id"]]["status"]
        for _, row in results
    }
    positions = [
        result_status[row["result_id"]]
        for seed in document["seeds"] for item in seed["lists"]
        for row in item["ranking"]
    ]
    result_counts, seed_counts = count(results), count(seeds)
    position_available = positions.count("available")
    report: Dict[str, Any] = {
        "schema_version": 11,
        "artifact_kind": "live_preview_resolution_audit",
        "served_lists_sha256": document["content_sha256"],
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "external_request": {
            "method": "GET",
            "query_parameters": ["id (numeric Deezer track ID only)"],
            "body": None,
            "credentials": "none",
            "ratings_rater_session_transmitted": False,
            "signed_preview_urls_persisted": False,
        },
        "unique_deezer_ids_requested": len(ids),
        "unique_results": result_counts,
        "seeds": seed_counts,
        "ranked_positions": {
            "total": len(positions),
            "available": position_available,
            "no_preview": positions.count("no_preview"),
            "errors": positions.count("error"),
            "resolvable_fraction": position_available / len(positions),
        },
        "no_preview_result_deezer_ids": sorted({
            row["deezer_track_id"] for _, row in results
            if outcomes[row["deezer_track_id"]]["status"] == "no_preview"
        }),
        "no_preview_seed_deezer_ids": sorted({
            row["deezer_track_id"] for _, row in seeds
            if outcomes[row["deezer_track_id"]]["status"] == "no_preview"
        }),
        "errors": [
            {"deezer_track_id": track_id, "error": outcome["error"]}
            for track_id, outcome in outcomes.items() if outcome["status"] == "error"
        ],
        "observed_cors": sorted({
            outcome["cors"] for outcome in outcomes.values() if outcome.get("cors")
        }),
        "observed_cache_control": sorted({
            outcome["cache_control"] for outcome in outcomes.values()
            if outcome.get("cache_control")
        }),
        "chrome_playback": {"status": "pending manual verification"},
    }
    report["content_sha256"] = content_hash(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-audio-access")
    freeze.add_argument(
        "--old-protocol", type=Path, default=DEFAULT_OLD_DIR / "protocol-v10.json"
    )
    freeze.add_argument(
        "--old-lists", type=Path, default=DEFAULT_OLD_DIR / "served-lists-v10.json"
    )
    freeze.add_argument("--catalog-index", type=Path, required=True)
    freeze.add_argument("--out-dir", type=Path, default=DEFAULT_NEW_DIR)
    freeze.add_argument(
        "--evaluator", type=Path,
        default=ROOT / "benchmarks" / "human_eval_v11.html",
    )
    local = sub.add_parser("serve")
    local.add_argument("--port", type=int, default=8000)
    local.add_argument("--directory", type=Path, default=DEFAULT_NEW_DIR)
    local.add_argument("--no-open", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("--lists", type=Path, default=DEFAULT_NEW_DIR / "served-lists-v11.json")
    audit.add_argument("--endpoint", default=PRODUCTION_PREVIEW_ENDPOINT)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--workers", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-audio-access":
        paths = freeze_audio_access_pack(
            args.old_protocol, args.old_lists, args.catalog_index,
            args.out_dir, args.evaluator,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        print("RANKINGS_LOCKED; metadata-only audio erratum signed; no ratings created.")
    elif args.command == "serve":
        serve(
            port=args.port, directory=args.directory,
            open_browser=not args.no_open,
        )
    else:
        report = audit_preview_coverage(args.lists, args.endpoint, workers=args.workers)
        _write(args.output, report)
        positions = report["ranked_positions"]
        print(
            f"{positions['available']}/{positions['total']} ranked positions "
            f"({positions['resolvable_fraction']:.1%}) resolve"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
