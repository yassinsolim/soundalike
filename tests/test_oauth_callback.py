"""Integration test for the local OAuth callback loop.

Exercises the real HTTP callback server end-to-end on loopback (no Spotify
contact): it starts the same server used during `soundalike login`, simulates
the browser redirect, and confirms the auth code is captured and the anti-CSRF
state is validated. This covers the part of the live flow that unit tests with
mocks cannot.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request

import pytest

from soundalike.spotify.auth import _wait_for_callback

REDIRECT = "http://127.0.0.1:8899/callback"


def _run_callback(redirect: str, state: str, sink: dict) -> threading.Thread:
    def run() -> None:
        try:
            sink["code"] = _wait_for_callback(redirect, state, timeout=10)
        except Exception as exc:  # noqa: BLE001 - captured for assertion
            sink["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _send_redirect(url: str) -> None:
    # The server binds inside its thread; retry briefly until it accepts.
    last_err = None
    for _ in range(50):
        try:
            urllib.request.urlopen(url, timeout=5).read()
            return
        except urllib.error.URLError as err:
            last_err = err
            time.sleep(0.1)
    raise AssertionError(f"Could not reach local callback server: {last_err}")


def test_callback_captures_code():
    sink: dict = {}
    thread = _run_callback(REDIRECT, "state-abc", sink)
    _send_redirect(f"{REDIRECT}?code=THE_CODE&state=state-abc")
    thread.join(10)
    assert sink.get("code") == "THE_CODE", sink


def test_callback_rejects_state_mismatch():
    sink: dict = {}
    thread = _run_callback(REDIRECT, "expected-state", sink)
    _send_redirect(f"{REDIRECT}?code=THE_CODE&state=attacker-state")
    thread.join(10)
    assert "code" not in sink
    assert isinstance(sink.get("error"), RuntimeError)


def test_callback_reports_provider_error():
    sink: dict = {}
    thread = _run_callback(REDIRECT, "s", sink)
    _send_redirect(f"{REDIRECT}?error=access_denied&state=s")
    thread.join(10)
    assert isinstance(sink.get("error"), RuntimeError)
