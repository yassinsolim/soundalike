"""On-disk cache of computed acoustic features.

Analyzing a preview costs a download + a few seconds of DSP, so we persist every
result keyed by a stable track id. Re-running a recommendation then reuses the
measured fingerprints instead of recomputing them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from ..config import cache_dir
from .features import AcousticFeatures


class FeatureStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or (cache_dir() / "acoustic_features.json")
        self._cache: Dict[str, AcousticFeatures] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._cache = {k: AcousticFeatures.from_dict(v) for k, v in raw.items()}
            except (ValueError, KeyError):
                self._cache = {}

    def save(self) -> None:
        payload = {k: v.to_dict() for k, v in self._cache.items()}
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def key(source: str, track_id) -> str:
        return f"{source}:{track_id}"

    def get(self, key: str) -> Optional[AcousticFeatures]:
        return self._cache.get(key)

    def put(self, key: str, features: AcousticFeatures) -> None:
        self._cache[key] = features

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)
