"""Keep test subprocesses bound to this checkout's src tree."""

from __future__ import annotations

import os
from pathlib import Path


SRC = str(Path(__file__).resolve().parents[1] / "src")
existing = os.environ.get("PYTHONPATH", "").split(os.pathsep)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [SRC, *(item for item in existing if item and item != SRC)]
)
