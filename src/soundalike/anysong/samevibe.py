"""Same vibe: Best matches that also share the seed's tempo and overall sound, one song per artist.

Best looks for shared material (covers, samples, melodies) and its weak picks can come from any genre. Same vibe keeps
the Best candidates a listener would hear as related: the first ``pool`` songs of the served Best order whose tempo is
compatible with the seed's (the same, double or half, within ``tempo_tol``) and that sound alike (in the seed's sounds
list, or a MuQ-MuLan score of at least ``tau``). The sounds list fills the rest, tempo-compatible songs first. Selection
and the one-time confirmation are in docs/ANYSONG_EVAL.md#same-vibe-2026-11-06.

Inputs follow ``api.AnySong._view``: the song view of the first 200 rows per channel without the seed's own song,
language-filtered in same-language mode. A candidate's fields come from its representative track: the item of the first
channel, in the order melody, samples, sonic, whose list holds the song.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

CHANNEL_ORDER = ("melody", "samples", "sonic")  # representative track: the first of these lists that holds the song


@dataclass(frozen=True)
class SameVibeParams:
    pool: int = 100  # Best songs considered for the gated part
    tempo_tol: float = 0.06
    # Sound gate: a served MuQ-MuLan score (maxsim minus the hub penalty) at least this high counts as sounding alike.
    # The median 800th sonic score of the development seeds on the 2026-11-01 index; recompute when the index changes.
    tau: float = 0.37819311022758484
    size: int = 20


@dataclass(frozen=True)
class Candidate:
    track_id: int
    bpm: float | None  # None or NaN = unknown
    muq: float  # served MuQ-MuLan score against the seed
    artist_id: int = 0  # 0 = unknown: the song counts as its own artist
    language: str = "unknown"


def tempo_compatible(seed_bpm: float | None, bpm: float | None, tol: float) -> bool:
    """Tempos within ``tol`` of each other, or of double or half; an unknown tempo is compatible with anything."""
    if seed_bpm is None or bpm is None:
        return True
    a, b = float(seed_bpm), float(bpm)
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
        return True
    ratio = a / b
    return min(abs(ratio / m - 1.0) for m in (1.0, 2.0, 0.5)) <= tol


def same_vibe(best: Sequence[Hashable], sonic: Sequence[Hashable], cands: Mapping[Hashable, Candidate],
              seed_bpm: float | None, seed_language: str = "unknown", mode: str = "same",
              params: SameVibeParams = SameVibeParams()) -> list:
    """Up to ``params.size`` song keys.

    1. The first ``pool`` songs of ``best`` (the served Best order) that pass the tempo and the sound gate.
    2. Then ``sonic`` songs (the seed's sounds list) that pass the tempo gate, then the remaining ``sonic`` songs, then
       the remaining ``best`` songs, each in its own order.

    Every step skips songs already taken and artists already present. In same-language mode a song whose language is
    known and differs from the seed's is never taken (the API's inputs are already filtered that way).
    """
    filter_language = mode == "same" and seed_language not in (None, "", "unknown")
    in_sonic = set(sonic)
    out: list = []
    taken: set = set()
    artists: set = set()

    def take(key: Hashable) -> None:
        if len(out) >= params.size or key in taken:
            return
        cand = cands[key]
        if filter_language and cand.language not in ("unknown", seed_language):
            return
        if cand.artist_id and cand.artist_id in artists:
            return
        out.append(key)
        taken.add(key)
        if cand.artist_id:
            artists.add(cand.artist_id)

    def tempo_ok(key: Hashable) -> bool:
        return tempo_compatible(seed_bpm, cands[key].bpm, params.tempo_tol)

    for key in best[:params.pool]:
        if tempo_ok(key) and (key in in_sonic or cands[key].muq >= params.tau):
            take(key)
    for key in sonic:
        if tempo_ok(key):
            take(key)
    for key in (*sonic, *best):
        take(key)
    return out
