#!/usr/bin/env python3
"""Stage, validate, activate, and roll back immutable Soundalike releases.

Run this as root on the VM. It deliberately accepts only a full Git commit SHA:
branches, tags, passwords, and SSH-host-checking bypasses are not deployment inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional
from urllib.parse import urlparse
from urllib.request import urlopen


DEFAULT_REPOSITORY = "https://github.com/yassinsolim/soundalike.git"
DEFAULT_ROOT = Path("/opt/soundalike")
DEFAULT_SERVICE = "soundalike.service"
DEFAULT_ORIGIN = "http://127.0.0.1:8788"
STARTUP_TIMEOUT_SECONDS = 120
STARTUP_RETRY_SECONDS = 2
MANIFEST_PATH = Path("deploy/homelab/release-manifest.json")
REQUIRED_RUNTIME_FILES = {
    "deploy/homelab/probe_v4.py",
    "deploy/homelab/update.py",
    "deploy/homelab/server.py",
    "deploy/homelab/soundalike.service",
    "webapp/api/_reco.py",
    "webapp/api/_search.py",
    "webapp/api/spicetify_recommend.py",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class UpdateError(RuntimeError):
    """A release could not safely be activated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise UpdateError(f"manifest path escapes release: {value}") from error
    return path


def _checksum(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise UpdateError(f"{label} must contain a lowercase SHA-256 checksum")
    return value


@dataclass(frozen=True)
class ReleaseManifest:
    runtime_files: Mapping[str, str]
    requirements_path: str
    requirements_sha256: str
    index_url: str
    index_sha256: str


def load_manifest(release: Path) -> ReleaseManifest:
    try:
        raw = json.loads(_safe_path(release, str(MANIFEST_PATH)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UpdateError(f"invalid release manifest: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema") != 1:
        raise UpdateError("release manifest must use schema 1")
    runtime_files = raw.get("runtime_files")
    if not isinstance(runtime_files, dict) or not REQUIRED_RUNTIME_FILES.issubset(runtime_files):
        raise UpdateError("release manifest does not cover every required runtime file")
    checked = {
        str(path): _checksum(checksum, f"runtime checksum for {path}")
        for path, checksum in runtime_files.items()
    }
    requirements = raw.get("requirements")
    if not isinstance(requirements, dict) or requirements.get("path") != "webapp/requirements.txt":
        raise UpdateError("release manifest must checksum webapp/requirements.txt")
    index = raw.get("index")
    if not isinstance(index, dict) or not isinstance(index.get("url"), str):
        raise UpdateError("release manifest must define an index URL")
    if urlparse(index["url"]).scheme != "https":
        raise UpdateError("release index URL must use HTTPS")
    return ReleaseManifest(
        runtime_files=checked,
        requirements_path=requirements["path"],
        requirements_sha256=_checksum(requirements.get("sha256"), "requirements checksum"),
        index_url=index["url"],
        index_sha256=_checksum(index.get("sha256"), "index checksum"),
    )


def verify_release(release: Path, target_commit: str, git_output: Callable[[Iterable[str]], str]) -> ReleaseManifest:
    if not COMMIT_RE.fullmatch(target_commit):
        raise UpdateError("target must be a full 40-character lowercase commit SHA")
    actual = git_output(("git", "-C", str(release), "rev-parse", "HEAD")).strip().lower()
    if actual != target_commit:
        raise UpdateError(f"staged commit is {actual}, not requested {target_commit}")
    manifest = load_manifest(release)
    for relative, expected in manifest.runtime_files.items():
        path = _safe_path(release, relative)
        if not path.is_file() or sha256(path) != expected:
            raise UpdateError(f"runtime checksum verification failed: {relative}")
    requirements = _safe_path(release, manifest.requirements_path)
    if not requirements.is_file() or sha256(requirements) != manifest.requirements_sha256:
        raise UpdateError("requirements checksum verification failed")
    return manifest


def download_index(manifest: ReleaseManifest, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".deepvibe_index.", dir=destination.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with urlopen(manifest.index_url, timeout=120) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(temporary_path) != manifest.index_sha256:
            raise UpdateError("index checksum verification failed")
        temporary_path.chmod(0o640)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def secure_release(release: Path) -> None:
    """Make a completed release readable, but not writable, by the service group."""
    import grp
    import stat

    try:
        group_id = grp.getgrnam("soundalike").gr_gid
    except KeyError as error:
        raise UpdateError("the soundalike system group must exist before deployment") from error
    for path in (release, *release.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            continue
        os.chown(path, -1, group_id)
        if stat.S_ISDIR(mode):
            path.chmod(0o750)
        elif mode & stat.S_IXUSR:
            path.chmod(0o750)
        else:
            path.chmod(0o640)


def _release_target(link: Path, releases: Path) -> Optional[Path]:
    if not link.is_symlink():
        return None
    target = link.resolve(strict=True)
    try:
        target.relative_to(releases.resolve())
    except ValueError as error:
        raise UpdateError(f"{link} points outside the release directory") from error
    return target


def atomic_link(link: Path, target: Path) -> None:
    temporary = link.with_name(f".{link.name}.new-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)


def switch_release(root: Path, release: Path) -> Optional[Path]:
    releases = root / "releases"
    previous = _release_target(root / "current", releases)
    atomic_link(root / "current", release)
    if previous is not None:
        atomic_link(root / "previous", previous)
    return previous


def restore_release(root: Path, previous: Path) -> None:
    atomic_link(root / "current", previous)
    atomic_link(root / "previous", previous)


def _run(command: Iterable[str], *, cwd: Optional[Path] = None, capture: bool = False) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return completed.stdout or ""


class Updater:
    def __init__(self, root: Path, repository: str, target: str, origin: str = DEFAULT_ORIGIN):
        if origin.rstrip("/") != DEFAULT_ORIGIN:
            raise UpdateError(f"deployment probes must use the local origin {DEFAULT_ORIGIN}")
        self.root = root
        self.repository = repository
        self.target = target
        self.origin = DEFAULT_ORIGIN
        self.releases = root / "releases"

    def dry_run(self) -> None:
        print(f"would stage commit {self.target} from {self.repository}")
        print(f"would verify {MANIFEST_PATH}, runtime files, requirements, and release index")
        print(f"would atomically switch {self.root / 'current'} and restart {DEFAULT_SERVICE}")
        print(f"would verify {self.origin}/healthz and the strict API v4 canary")

    def _stage(self) -> Path:
        release = self.releases / self.target
        if release.exists():
            raise UpdateError(f"immutable release already exists: {release}")
        self.releases.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{self.target}.", dir=self.releases))
        try:
            _run(("git", "clone", "--no-checkout", self.repository, str(staging)))
            _run(("git", "-C", str(staging), "config", "core.autocrlf", "false"))
            _run(("git", "-C", str(staging), "fetch", "--depth=1", "origin", self.target))
            _run(("git", "-C", str(staging), "checkout", "--detach", "--force", self.target))
            manifest = verify_release(
                staging, self.target,
                lambda command: _run(command, capture=True),
            )
            shutil.rmtree(staging / ".git")
            _run((sys.executable, "-m", "venv", str(staging / ".venv")))
            _run((
                str(staging / ".venv/bin/pip"), "install", "--disable-pip-version-check",
                "--no-input", "-r", str(_safe_path(staging, manifest.requirements_path)),
            ))
            download_index(manifest, staging / "runtime/deepvibe_index.npz")
            secure_release(staging)
            os.replace(staging, release)
            return release
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _install_unit(self, release: Path) -> None:
        source = release / "deploy/homelab/soundalike.service"
        destination = Path("/etc/systemd/system") / DEFAULT_SERVICE
        temporary = destination.with_name(f".{destination.name}.new-{os.getpid()}")
        shutil.copyfile(source, temporary)
        temporary.chmod(0o644)
        os.replace(temporary, destination)
        _run(("systemctl", "daemon-reload"))

    def _restart_and_probe(self, release: Path, health_only: bool = False) -> None:
        _run(("systemctl", "restart", DEFAULT_SERVICE))
        probe = (
            str(release / ".venv/bin/python"),
            str(release / "deploy/homelab/probe_v4.py"),
            "--url", self.origin,
        )
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            try:
                _run((*probe, "--health-only"))
                break
            except subprocess.CalledProcessError:
                if time.monotonic() >= deadline:
                    raise UpdateError(
                        f"{DEFAULT_SERVICE} did not become healthy within "
                        f"{STARTUP_TIMEOUT_SECONDS} seconds"
                    )
                time.sleep(STARTUP_RETRY_SECONDS)
        if not health_only:
            _run(probe)

    def execute(self) -> None:
        release = self._stage()
        previous = _release_target(self.root / "current", self.releases)
        try:
            switch_release(self.root, release)
            self._install_unit(release)
            self._restart_and_probe(release)
        except Exception as failure:
            if previous is None:
                raise UpdateError(f"activation failed with no release to roll back to: {failure}") from failure
            try:
                restore_release(self.root, previous)
                self._install_unit(previous)
                self._restart_and_probe(previous, health_only=True)
            except Exception as rollback:
                raise UpdateError(
                    f"activation failed ({failure}); rollback health verification failed ({rollback})"
                ) from rollback
            raise UpdateError(f"activation failed and was rolled back: {failure}") from failure
        print(f"activated immutable release {release.name}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="Full immutable 40-character Git commit SHA")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Release root (VM default: /opt/soundalike)")
    parser.add_argument("--dry-run", action="store_true", help="Print validation and activation steps without changes")
    args = parser.parse_args(argv)
    target = args.commit.lower()
    if not COMMIT_RE.fullmatch(target):
        parser.error("--commit must be a full 40-character Git commit SHA")
    updater = Updater(args.root, DEFAULT_REPOSITORY, target)
    if args.dry_run:
        updater.dry_run()
        return 0
    if os.name != "posix" or os.geteuid() != 0:
        parser.error("run as root on the Linux VM (use sudo; no password is accepted as an option)")
    try:
        updater.execute()
    except (UpdateError, subprocess.CalledProcessError, OSError) as error:
        print(f"update failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
