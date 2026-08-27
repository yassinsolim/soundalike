#!/usr/bin/env python3
"""Write or verify the checksummed homelab release manifest before committing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "deploy/homelab/release-manifest.json"
RUNTIME_FILES = (
    "deploy/homelab/probe_v4.py",
    "deploy/homelab/update.py",
    "deploy/homelab/server.py",
    "deploy/homelab/soundalike.service",
    "webapp/api/_reco.py",
    "webapp/api/_search.py",
    "webapp/api/spicetify_recommend.py",
)
INDEX_URL = (
    "https://github.com/yassinsolim/soundalike/releases/download/"
    "index-2026.07.11-dual-sonic64/deepvibe_index.npz"
)
INDEX_SHA256 = "f3ed57af1b8073f2872eed1e9192dee04d1089c7266fb98a157d1ea194526fb9"


def digest(path: Path) -> str:
    # Releases explicitly check out with core.autocrlf=false, so hash those LF bytes
    # even when a contributor's Windows working tree uses CRLF.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def content() -> str:
    manifest = {
        "schema": 1,
        "runtime_files": {path: digest(ROOT / path) for path in RUNTIME_FILES},
        "requirements": {
            "path": "webapp/requirements.txt",
            "sha256": digest(ROOT / "webapp/requirements.txt"),
        },
        "index": {"url": INDEX_URL, "sha256": INDEX_SHA256},
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail unless the committed manifest is current")
    args = parser.parse_args(argv)
    expected = content()
    if args.check:
        if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != expected:
            print("release manifest is stale; run deploy/homelab/write_manifest.py", file=sys.stderr)
            return 1
        return 0
    MANIFEST.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
