#!/usr/bin/env python3
"""Fail-closed health and API-v4 canary for the always-on Soundalike origin."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_ORIGIN = "http://127.0.0.1:8788"
EXPECTED_METHOD = "dual_sonic64_guardrail"
EXPECTED_INDEX_VERSION = "2026.07.11-dual-sonic64"
EXPECTED_LANGUAGE_POLICY = "spotify-lyrics-strict-v2"
DEFAULT_SEED = "Redbone — Childish Gambino"


class ProbeError(RuntimeError):
    """The endpoint did not meet the production compatibility contract."""


def canary_url(origin: str, seed: str = DEFAULT_SEED) -> str:
    origin = origin.rstrip("/")
    if not origin.startswith(("http://", "https://")):
        raise ProbeError("origin must use http or https")
    return f"{origin}/api/spicetify_recommend?{urlencode({
        'query': seed,
        'n': '3',
        'diversity': '0.15',
        'v': '4',
        'language_policy': EXPECTED_LANGUAGE_POLICY,
    })}"


def _read_json(
    url: str, timeout: float, opener: Callable[..., Any] = urlopen
) -> Dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Soundalike-V4-Monitor/1.0",
        },
    )
    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if status != 200:
                raise ProbeError(f"{url} returned HTTP {status}")
            raw = response.read()
    except HTTPError as error:
        raise ProbeError(f"{url} returned HTTP {error.code}") from error
    except URLError as error:
        raise ProbeError(f"could not reach {url}: {error.reason}") from error
    except OSError as error:
        raise ProbeError(f"could not reach {url}: {error}") from error

    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError(f"{url} returned malformed JSON") from error
    if not isinstance(body, dict):
        raise ProbeError(f"{url} returned a JSON value instead of an object")
    return body


def check_health(
    origin: str, timeout: float = 15, opener: Callable[..., Any] = urlopen
) -> None:
    body = _read_json(f"{origin.rstrip('/')}/healthz", timeout, opener)
    if body != {"ok": True}:
        raise ProbeError("health endpoint did not return exactly {'ok': true}")


def check_v4(
    origin: str,
    seed: str = DEFAULT_SEED,
    timeout: float = 15,
    opener: Callable[..., Any] = urlopen,
) -> None:
    body = _read_json(canary_url(origin, seed), timeout, opener)
    if body.get("ok") is not True:
        raise ProbeError("v4 canary did not report ok")
    if body.get("method") != EXPECTED_METHOD:
        raise ProbeError("v4 canary returned an unexpected ranking method")
    if body.get("retrieval_mode") != EXPECTED_METHOD:
        raise ProbeError("v4 canary returned an unexpected retrieval mode")
    if body.get("index_version") != EXPECTED_INDEX_VERSION:
        raise ProbeError("v4 canary returned an unexpected index version")
    if body.get("language_policy") != EXPECTED_LANGUAGE_POLICY:
        raise ProbeError("v4 canary returned an unexpected language policy")
    if not isinstance(body.get("results"), list) or not body["results"]:
        raise ProbeError("v4 canary returned no recommendations")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_ORIGIN, help="Origin URL (default: local tunnel target)")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Known catalog seed for the v4 canary")
    parser.add_argument("--timeout", type=float, default=15, help="Request timeout in seconds")
    parser.add_argument("--health-only", action="store_true", help="Check /healthz without querying the model")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        check_health(args.url, args.timeout)
        if not args.health_only:
            check_v4(args.url, args.seed, args.timeout)
    except ProbeError as error:
        print(f"probe failed: {error}", file=sys.stderr)
        return 1
    print("probe passed: health and v4 contract" if not args.health_only else "probe passed: health")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
