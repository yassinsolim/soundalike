import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    path = ROOT / "deploy" / "homelab" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"homelab_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release(updater, tmp_path, commit):
    release = tmp_path / "release"
    for relative in updater.REQUIRED_RUNTIME_FILES:
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"runtime {relative}", encoding="utf-8")
    requirements = release / "webapp/requirements.txt"
    requirements.parent.mkdir(parents=True, exist_ok=True)
    requirements.write_text("numpy==1.0\n", encoding="utf-8")
    runtime_files = {
        relative: _sha256(release / relative)
        for relative in updater.REQUIRED_RUNTIME_FILES
    }
    manifest = {
        "schema": 1,
        "runtime_files": runtime_files,
        "requirements": {"path": "webapp/requirements.txt", "sha256": _sha256(requirements)},
        "index": {
            "url": "https://example.test/deepvibe_index.npz",
            "sha256": "0" * 64,
        },
    }
    path = release / updater.MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return release


def test_release_verification_refuses_a_changed_requirement_checksum(tmp_path):
    updater = _module("update")
    commit = "a" * 40
    release = _release(updater, tmp_path, commit)
    updater.verify_release(release, commit, lambda _command: f"{commit}\n")

    (release / "webapp/requirements.txt").write_text("numpy==9.9\n", encoding="utf-8")

    with pytest.raises(updater.UpdateError, match="requirements checksum"):
        updater.verify_release(release, commit, lambda _command: f"{commit}\n")


@pytest.mark.skipif(os.name != "posix", reason="atomic symlinks require the Linux deployment host")
def test_successful_switch_is_atomic_and_preserves_previous_release(tmp_path):
    updater = _module("update")
    root = tmp_path / "soundalike"
    old = root / "releases/old"
    new = root / "releases/new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    updater.atomic_link(root / "current", old)

    previous = updater.switch_release(root, new)

    assert previous == old
    assert (root / "current").resolve() == new
    assert (root / "previous").resolve() == old


@pytest.mark.skipif(os.name != "posix", reason="atomic symlinks require the Linux deployment host")
def test_failed_activation_rolls_back_and_verifies_old_health(tmp_path, monkeypatch):
    updater = _module("update")
    root = tmp_path / "soundalike"
    old = root / "releases/old"
    new = root / "releases/new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    updater.atomic_link(root / "current", old)
    deployment = updater.Updater(root, "https://example.test/repo.git", "b" * 40, "http://127.0.0.1:8788")
    calls = []
    monkeypatch.setattr(deployment, "_stage", lambda: new)
    monkeypatch.setattr(deployment, "_install_unit", lambda release: calls.append(("unit", release.name)))

    def probe(release, health_only=False):
        calls.append(("probe", release.name, health_only))
        if release == new:
            raise RuntimeError("canary failed")

    monkeypatch.setattr(deployment, "_restart_and_probe", probe)

    with pytest.raises(updater.UpdateError, match="was rolled back"):
        deployment.execute()

    assert (root / "current").resolve() == old
    assert ("probe", old.name, True) in calls


def test_restart_waits_for_model_startup_before_full_canary(tmp_path, monkeypatch):
    updater = _module("update")
    release = tmp_path / "release"
    python = release / ".venv/bin/python"
    probe = release / "deploy/homelab/probe_v4.py"
    calls = []
    attempts = iter([False, False, True])

    def run(command, **_kwargs):
        command = tuple(command)
        calls.append(command)
        if command == (str(python), str(probe), "--url", updater.DEFAULT_ORIGIN,
                       "--health-only") and not next(attempts):
            raise subprocess.CalledProcessError(1, command)
        return ""

    monkeypatch.setattr(updater, "_run", run)
    monkeypatch.setattr(updater.time, "sleep", lambda _seconds: None)
    deployment = updater.Updater(
        tmp_path / "soundalike",
        "https://example.test/repo.git",
        "b" * 40,
    )

    deployment._restart_and_probe(release)

    health = (str(python), str(probe), "--url", updater.DEFAULT_ORIGIN,
              "--health-only")
    full = (str(python), str(probe), "--url", updater.DEFAULT_ORIGIN)
    assert calls.count(health) == 3
    assert calls[-1] == full


def test_monitor_workflow_checks_secrets_inside_a_step():
    workflow = (ROOT / ".github/workflows/api-monitor.yml").read_text(
        encoding="utf-8"
    )
    job_if = workflow.split("  deploy:", 1)[1].split("    needs:", 1)[0]
    assert "secrets." not in job_if
    assert "Protected deployment secrets are incomplete" in workflow


class _Response:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def test_v4_probe_accepts_the_complete_contract_and_local_port():
    probe = _module("probe_v4")
    response = {
        "ok": True,
        "method": probe.EXPECTED_METHOD,
        "retrieval_mode": probe.EXPECTED_METHOD,
        "index_version": probe.EXPECTED_INDEX_VERSION,
        "language_policy": probe.EXPECTED_LANGUAGE_POLICY,
        "results": [{"title": "Candidate"}],
    }
    requested = []

    def opener(request, timeout):
        requested.append((request, timeout))
        return _Response(200, json.dumps(response).encode())

    probe.check_v4(probe.DEFAULT_ORIGIN, opener=opener)

    requested_url = requested[0][0].full_url
    assert requested_url.startswith("http://127.0.0.1:8788/api/spicetify_recommend?")
    assert "v=4" in requested_url
    assert "language_policy=spotify-lyrics-strict-v2" in requested_url
    assert requested[0][0].get_header("User-agent") == "Soundalike-V4-Monitor/1.0"
    service = (ROOT / "deploy/homelab/soundalike.service").read_text(encoding="utf-8")
    assert "Environment=SOUNDALIKE_HOST=127.0.0.1" in service
    assert "Environment=SOUNDALIKE_PORT=8788" in service
    assert "/opt/soundalike/current/" in service


@pytest.mark.parametrize(
    "opener, message",
    [
        (lambda _url, timeout: _Response(502, b"{}"), "HTTP 502"),
        (lambda _url, timeout: _Response(200, b"not json"), "malformed JSON"),
    ],
)
def test_v4_probe_refuses_gateway_errors_and_malformed_json(opener, message):
    probe = _module("probe_v4")
    with pytest.raises(probe.ProbeError, match=message):
        probe.check_v4(probe.DEFAULT_ORIGIN, opener=opener)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("method", "legacy", "ranking method"),
        ("retrieval_mode", "legacy", "retrieval mode"),
        ("index_version", "wrong-index", "index version"),
        ("language_policy", "permissive", "language policy"),
        ("results", [], "no recommendations"),
    ],
)
def test_v4_probe_refuses_wrong_ranking_contract(field, value, message):
    probe = _module("probe_v4")
    response = {
        "ok": True,
        "method": probe.EXPECTED_METHOD,
        "retrieval_mode": probe.EXPECTED_METHOD,
        "index_version": probe.EXPECTED_INDEX_VERSION,
        "language_policy": probe.EXPECTED_LANGUAGE_POLICY,
        "results": [{"title": "Candidate"}],
    }
    response[field] = value
    with pytest.raises(probe.ProbeError, match=message):
        probe.check_v4(
            probe.DEFAULT_ORIGIN,
            opener=lambda _url, timeout: _Response(200, json.dumps(response).encode()),
        )
