"""Fetch the deep-vibe pack (encoder + song index) from a GitHub Release.

The bundled index is capped by GitHub's 100 MB per-file limit, so a large
library (hundreds of thousands of songs, or a higher-dimensional embedding)
can't live in the repo. This module lets the pack live on a **GitHub Release**
instead — releases allow up to 2 GB per asset, are free, and don't bloat the
repo or every clone.

Resolution order for the pack, so it "just works" and stays offline-friendly:

  1. an explicit path the user passed (``--index`` / ``--model-dir``);
  2. a copy already downloaded to the user cache that matches the manifest;
  3. the copy **bundled** with the package, if it matches the manifest
     (so a fresh install with the small bundled library needs no download);
  4. otherwise **download** the manifest's pack from the Release, verify its
     SHA-256, and cache it — with a graceful fallback to the bundled copy if
     the download fails (e.g. offline).

`data/index_manifest.json` (tiny, tracked in the repo) is the single source of
truth for which pack is canonical; growing the library later is just re-uploading
assets and bumping that manifest.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from ..config import cache_dir


def _manifest_path() -> Optional[Path]:
    try:
        from importlib import resources

        res = resources.files("soundalike").joinpath("data/index_manifest.json")
        with resources.as_file(res) as p:
            if Path(p).exists():
                return Path(p)
    except (ModuleNotFoundError, FileNotFoundError, AttributeError, TypeError):
        pass
    local = Path(__file__).resolve().parents[1] / "data" / "index_manifest.json"
    return local if local.exists() else None


def load_manifest() -> Optional[Dict]:
    p = _manifest_path()
    if p is None:
        return None
    return json.loads(p.read_text())


def _bundled(name: str) -> Optional[Path]:
    p = Path(__file__).resolve().parents[1] / "data" / name
    return p if p.exists() else None


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _asset_url(manifest: Dict, asset: str) -> str:
    repo = manifest["repo"]
    tag = manifest["release_tag"]
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def _download(url: str, dest: Path, expected_sha: Optional[str],
              progress: Callable[[str], None]) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        progress(f"Downloading {dest.name} from {url} ...")
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length", 0))
            got = 0
            h = hashlib.sha256()
            with open(tmp, "wb") as out:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    h.update(block)
                    got += len(block)
                    if total:
                        pct = 100 * got / total
                        if got % (16 << 20) < (1 << 20):
                            progress(f"  {got/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)")
        if expected_sha and h.hexdigest() != expected_sha:
            progress(f"  checksum mismatch for {dest.name}; discarding download.")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        progress(f"  saved -> {dest}")
        return True
    except Exception as exc:  # noqa: BLE001
        progress(f"  download failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def _resolve_one(spec: Dict, manifest: Dict, cache: Path,
                 force: bool, progress: Callable[[str], None]) -> Optional[Path]:
    """Resolve a single asset (encoder or index) to a local path."""
    name = spec["asset"]
    sha = spec.get("sha256")

    cached = cache / name
    if not force and cached.exists() and (sha is None or _sha256(cached) == sha):
        return cached

    if not force:
        b = _bundled(name)
        if b is not None and (sha is None or _sha256(b) == sha):
            return b

    if _download(_asset_url(manifest, name), cached, sha, progress):
        return cached

    # Fallback: any bundled copy, even if it predates the manifest (offline).
    b = _bundled(name)
    if b is not None:
        progress(f"  using bundled {name} as a fallback.")
        return b
    return None


def ensure_pack(force: bool = False, progress: Callable[[str], None] = print
                ) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (encoder_path, index_path), downloading from the Release if needed.

    Either may be ``None`` if unavailable. Falls back to bundled copies offline.
    """
    manifest = load_manifest()
    if manifest is None:
        # No manifest: rely purely on whatever is bundled.
        return _bundled("vibe_encoder.pt"), _bundled("deepvibe_index.npz")

    cache = cache_dir() / "pack"
    enc = _resolve_one(manifest["encoder"], manifest, cache, force, progress)
    idx = _resolve_one(manifest["index"], manifest, cache, force, progress)
    return enc, idx


def describe() -> str:
    m = load_manifest()
    if m is None:
        return "No index manifest bundled; using bundled artifacts only."
    idx, enc = m["index"], m["encoder"]
    return (f"Pack '{m.get('version', '?')}' (release {m['release_tag']}): "
            f"{idx.get('n_tracks', '?'):,} tracks, {enc.get('embedding_dim','?')}-d encoder "
            f"({idx.get('bytes',0)/1e6:.0f} MB index).")
