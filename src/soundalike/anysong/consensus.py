"""Consensus body of the main list: songs that several views of the seed agree on, ordered by a small tree model
(docs/ANYSONG_EVAL.md, "Consensus body of the main list").

Five signals score every song of the catalog against the seed, each on its own scale, in sonic-index row order:

- ``snd``: the best window-pair cosine between the whitened sonic windows (the sonic index's ``full.npy``, unit
  length again in fp16), multiplied in fp16;
- ``head``: the same between scene-head vectors (the blend's table; a live seed's vectors come from its own windows);
- ``cf``: the cosine of playlist vectors, only when both songs are in the playlist graph (the blend's graph; the
  vectors are brought to unit length here as the harness does it, which differs from the blend's for the graph
  songs whose vector is all but zero);
- ``mel`` and ``smp``: the melody and samples indexes' stage-1 scores less the hub penalty (the searches' ``base``).

A song's rank under a signal counts the allowed songs that score higher, plus one. Allowed are the songs of the
seed's language or of unknown language (every song when the seed's language is unknown or every language is asked
for), without the songs of the seed's own artist. An artist is, for this body, the primary credited artist name: the
text before the first comma, lower-cased and stripped (not the scene table's artist id, which the lists' one song per
artist rule uses).

Candidates are the best ``K`` songs per signal and the best ``KC`` by consensus, the mean over the signals of
log10(rank) (``NO_RANK`` where a signal has no score for the song). Each candidate gets ``FEATURES``, a
``listener.Regressor`` predicts its 0 to 2 rating, and

    body score = prediction - sound_weight * min(log10(sound rank), 5)

orders the candidates whose sound rank is at most ``sound_cap``, one song per artist.

The arithmetic follows the study harness that fitted the model (anysong-lab v8/v12/pool.py and fit2.py) step by
step, dtypes included, because the model's splits sit on those values.

    python -m soundalike.anysong.consensus r3.pkl consensus-r3.npz      # export the fitted model
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import listener
from .scene import SceneParams, SceneTable, affinity, components

SIGNALS = ("snd", "head", "cf", "mel", "smp")
SIGNAL_NAMES = {"snd": "sound", "head": "head", "cf": "playlists", "mel": "melody", "smp": "samples"}
CHANNELS = {"mel": "melody", "smp": "samples"}  # signal -> the engine channel whose stage-1 scores it is
FEATURES = tuple(f"{p}_{s}" for s in SIGNALS for p in ("s", "z", "lr", "g1", "g10")) + (
    "n10", "n50", "n200", "n1000", "lr_min", "lr_2nd", "lr_mean", "tempo_fold", "tempo_strict", "lang_same",
    "lang_unknown", "lang_q_known", "graph", "region", "genre", "aff", "fwd", "rev", "nb", "cf_c", "cf_q")
K = 150  # best songs per signal
KC = 1500  # best songs by consensus
NO_RANK = 6.3  # log10(rank) of a song a signal has no score for
SOUND_WEIGHT = 0.1
SOUND_CAP = 5000
KEEP = 200  # songs of a body kept per seed (one per artist), enough for any list length after the lists' filters
CHUNK = 131072  # rows per fp16 product, as in the harness


def primary_artist(name: str) -> str:
    """The artist identity of this body: the credited name before the first comma, lower-cased and stripped."""
    return name.split(",")[0].strip().lower()


@dataclass(frozen=True)
class Seed:
    """What the features read of the seed besides its scores: its sonic-index row (None for a live seed), the index
    of its primary artist among the catalog's (None when the catalog has no song of it), its scene-table artist id,
    its tempo and its language code (-1: unknown, or every language asked for)."""

    row: int | None
    artist: int | None
    artist_id: int | None
    bpm: float
    code: int


@dataclass(frozen=True)
class Candidates:
    """Scored candidates of one seed: sonic-index ``rows`` (ascending), their ``FEATURES`` [C, F] float32, each
    signal's rank [C, 5] (0 without a score) and the body ``score`` (-inf past the sound cap)."""

    rows: np.ndarray
    X: np.ndarray
    ranks: np.ndarray
    score: np.ndarray


class Consensus:
    """The signals' tables next to the engine's, and the body they give a seed (``bodies``).

    ``cf``: the playlist vectors of the blend's graph songs, as stored (``cf128.npy``). ``windows``: the sonic
    index's full windows [N, W, D]; they stay in host memory (page-locked for a CUDA index) and
    cross to the index device in blocks for each seed, since the card has no room to hold them next to the engine.
    ``orders``: channel -> the track ids of the melody and samples indexes in their row order. ``artists``: the
    credited artist of every sonic row (None: unknown, an artist of its own). ``codes``: the gate's language code of
    every row (``languages.LanguageRows``; None: all unknown). ``bpm``: the catalog tempo of every row (NaN: none)."""

    def __init__(self, model: listener.Regressor, ids: np.ndarray, windows: np.ndarray, blend, cf: np.ndarray,
                 orders: Mapping[str, np.ndarray], artists: Sequence[str | None], codes=None,
                 bpm: np.ndarray | None = None, scene: SceneTable | None = None,
                 scene_params: SceneParams | None = None, prep=None, sound_weight: float = SOUND_WEIGHT,
                 sound_cap: int = SOUND_CAP):
        missing = [f for f in model.features if f not in FEATURES]
        if missing:
            raise ValueError(f"the consensus model reads features this code does not compute: {missing}")
        self.model, self.columns = model, [FEATURES.index(f) for f in model.features]
        self.ids = np.asarray(ids, dtype=np.int64)
        n = self.n = len(self.ids)
        if len(blend) != n or len(windows) != n or len(artists) != n:
            raise ValueError(f"the blend ({len(blend)} songs), the sonic windows ({len(windows)}) and the artists "
                             f"({len(artists)}) must cover the sonic index ({n} songs)")
        if sound_weight < 0 or sound_cap < 1:
            raise ValueError("the sound weight cannot be negative and the sound cap is a rank of at least 1")
        self.blend, self.prep = blend, prep
        self.sound_weight, self.sound_cap = float(sound_weight), int(sound_cap)
        self.scene, self.scene_params = scene, scene_params or SceneParams()
        self.device = device = blend.table.device
        # The model's arrays on the index device (``_predict``).
        self.trees = {name: torch.as_tensor(getattr(model, name), device=device)
                      for name in ("roots", "feature", "threshold", "missing_left", "left", "right", "value")}
        # The scene table's artist id of every row (0: none), as ``SceneTable.artist`` gives it.
        self.scene_artist = np.zeros(n, np.int64)
        if scene is not None and len(scene.track_ids):
            at = np.clip(np.searchsorted(scene.track_ids, self.ids), 0, len(scene.track_ids) - 1)
            self.scene_artist = np.where(scene.track_ids[at] == self.ids,
                                         np.maximum(scene.track_artist[at], 0), 0).astype(np.int64)
        # Unit windows in fp16, normalised on the index device as the harness does it.
        self.windows = torch.empty(tuple(windows.shape), dtype=torch.float16, pin_memory=device.type == "cuda")
        for s in range(0, n, 16384):
            x = torch.from_numpy(np.array(windows[s:s + 16384])).to(device).half()
            self.windows[s:s + 16384] = self._unit(x).cpu()
        # Primary artist name -> its number; a song without a credited artist is an artist of its own.
        self.artist_index: dict[str, int] = {}
        self.artist_np = np.array([self.artist_index.setdefault(primary_artist(a) if a is not None else f"?{i}",
                                                                 len(self.artist_index))
                                   for i, a in enumerate(artists)], dtype=np.int64)
        self.artist = torch.from_numpy(self.artist_np).to(device)
        self.codes = torch.full((n,), -1, dtype=torch.int16, device=device) if codes is None else \
            torch.as_tensor(codes, device=device)
        self.codes_np = self.codes.cpu().numpy().astype(np.int64)
        self.bpm = np.full(n, np.nan, np.float32) if bpm is None else np.asarray(bpm, dtype=np.float32)
        if self.codes.shape != (n,) or self.bpm.shape != (n,):
            raise ValueError(f"language codes and tempos must cover the sonic index ({n} songs)")
        self.graph_np = (blend.graph_pos >= 0).cpu().numpy()
        if np.ndim(cf) != 2 or len(cf) != len(blend.cf_rows):
            raise ValueError(f"playlist vectors {np.shape(cf)} do not fit the blend's {len(blend.cf_rows)} graph songs")
        cf = torch.from_numpy(np.asarray(cf, dtype=np.float32)).to(device)
        self.cf = cf / cf.norm(dim=1, keepdim=True).clamp_min(1e-9)
        # Sonic row -> row of the melody and samples indexes; None where the index shares the sonic row order.
        self.order: dict[str, torch.Tensor | None] = {}
        for key, channel in CHANNELS.items():
            other = np.asarray(orders[channel], dtype=np.int64)
            if np.array_equal(other, self.ids):
                self.order[key] = None
                continue
            at = np.argsort(other, kind="stable")
            found = at[np.clip(np.searchsorted(other[at], self.ids), 0, len(other) - 1)] if len(other) else at
            if len(other) != n or not np.array_equal(other[found], self.ids):
                raise ValueError(f"the {channel} index does not hold the sonic index's songs")
            self.order[key] = torch.from_numpy(found).to(device)
        self.log_rank = torch.arange(1, n + 1, device=device).float().log10()

    @classmethod
    def load(cls, model: Path | str, windows: Path | str, cf: Path | str, engine, **kw) -> Consensus:
        """``model``: an exported ``listener.Regressor``; ``windows``: the sonic index's ``full.npy``; ``cf``: the
        blend's ``cf128.npy``; ``engine``: a ``serve.Engine`` with a blend and the melody, samples and sonic indexes."""
        sonic = engine.indexes["sonic"]
        return cls(listener.Regressor.load(model), sonic.ids, np.load(windows, mmap_mode="r"), engine.blend,
                   np.load(cf), {channel: engine.indexes[channel].ids for channel in CHANNELS.values()}, prep=sonic.prep_query,
                   **kw)

    def info(self) -> dict:
        return {"model": self.model.meta or "on", "features": len(self.model.features),
                "sound_weight": self.sound_weight, "sound_cap": self.sound_cap}

    @staticmethod
    def _unit(x: torch.Tensor) -> torch.Tensor:
        return x / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).half()

    def _maxsim(self, q: torch.Tensor, table: torch.Tensor, unit: bool) -> torch.Tensor:
        """Best window-pair product of the seed's vectors [W, D] with every row's [N], multiplied in fp16. ``unit``:
        the table's rows are brought to unit length first (the head table, which is on the device as stored)."""
        out = torch.empty(self.n, device=self.device)
        qf = q.half()
        for s in range(0, self.n, CHUNK):
            block = table[s:s + CHUNK].to(self.device, non_blocking=True)
            if unit:
                block = self._unit(block.half())
            n, w, d = block.shape
            out[s:s + n] = (qf @ block.reshape(n * w, d).T).reshape(len(qf), n, w).amax(dim=(0, 2)).float()
        return out

    @torch.inference_mode()
    def signals(self, bases: Mapping[str, torch.Tensor], row: int | None = None,
                windows: np.ndarray | None = None) -> dict[str, torch.Tensor]:
        """Each signal's score of every sonic row [N] for a stored seed (``row``) or a live one (its raw sonic
        ``windows``), -inf where the signal has none. ``bases``: channel -> the stage-1 scores less the hub penalty
        of the seed's melody and samples searches, in those indexes' row order."""
        if row is not None:
            sound = self.windows[row].to(self.device)
            head = self._unit(self.blend.table[row].half())
        else:
            sound = self.prep(windows)
            head = self.blend.seed_head(None, windows)
        out = {"snd": self._maxsim(sound, self.windows, False), "head": self._maxsim(head, self.blend.table, True)}
        cf = torch.full((self.n,), -torch.inf, device=self.device)
        at = int(self.blend.graph_pos[row]) if row is not None else -1
        if at >= 0:
            cf[self.blend.cf_rows] = self.cf @ self.cf[at]
        out["cf"] = cf
        for key, channel in CHANNELS.items():
            base, order = bases[channel].float(), self.order[key]
            out[key] = base if order is None else base[order]
        return out

    @torch.inference_mode()
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """``model.predict`` (``listener.Regressor``) with the descent through the trees on the index device: the same
        comparisons in float64, and the leaf values summed tree by tree on the host, so the same numbers."""
        t = self.trees
        x = torch.from_numpy(X.astype(np.float64)).to(self.device)
        node = t["roots"].expand(len(x), -1).clone()
        rows = torch.arange(len(x), device=self.device)[:, None]
        for _ in range(self.model.depth):
            v = x[rows, t["feature"][node]]
            left = torch.where(torch.isnan(v), t["missing_left"][node], v <= t["threshold"][node])
            node = torch.where(left, t["left"][node], t["right"][node])
        out = np.full(len(x), self.model.baseline)
        for values in t["value"][node].cpu().numpy().T:
            out += values
        return out

    def seed(self, code: int, row: int | None = None, artist: str | None = None, artist_id: int | None = None,
             bpm: float | None = None) -> Seed:
        """The ``Seed`` of a stored song (``row``; the catalog says the rest) or of a live one."""
        if row is not None:
            artist_id = int(self.scene_artist[row]) or None
            return Seed(row, int(self.artist_np[row]), artist_id, float(self.bpm[row]), int(code))
        return Seed(None, self.artist_index.get(primary_artist(artist)) if artist else None, artist_id,
                    float(bpm) if bpm else float("nan"), int(code))

    def allowed(self, seed: Seed) -> torch.Tensor:
        """The rows a seed's ranks count: its language's songs and those of unknown language, without its artist's."""
        if seed.code >= 0:
            allow = (self.codes == seed.code) | (self.codes < 0)
        else:
            allow = torch.ones(self.n, dtype=torch.bool, device=self.device)
        if seed.artist is not None:
            allow &= self.artist != seed.artist
        if seed.row is not None:
            allow[seed.row] = False
        return allow

    @torch.inference_mode()
    def candidates(self, signals: Mapping[str, torch.Tensor], seed: Seed, rows: np.ndarray | None = None,
                   scene_cache: dict | None = None) -> Candidates:
        """The seed's candidates with their features and body scores; ``rows`` (a study's own sonic rows) replaces
        the pool. ``scene_cache``: the scene columns per artist, shared by the calls for one seed."""
        n, allow = self.n, self.allowed(seed)
        scores, ascending, stats, parts = {}, {}, {}, []
        consensus = None
        for key in SIGNALS:
            sc = signals[key].masked_fill(~allow, -torch.inf)
            scores[key] = sc
            ordered = sc.sort(descending=True)
            ascending[key] = srt = ordered.values.flip(0)
            finite = torch.isfinite(sc)
            count = finite.sum().clamp_min(1)
            mean = torch.where(finite, sc, torch.zeros_like(sc)).sum() / count
            sd = ((torch.where(finite, sc - mean, torch.zeros_like(sc)) ** 2).sum() / count).sqrt().clamp_min(1e-6)
            stats[key] = (mean, sd, srt[-1], srt[-min(10, n)])
            top = sc.topk(min(K, n)).indices
            parts.append(top[torch.isfinite(sc[top])])
            log_rank = torch.empty_like(sc)
            log_rank[ordered.indices] = self.log_rank
            log_rank[~finite] = NO_RANK
            consensus = log_rank if consensus is None else consensus + log_rank
        if rows is None:
            best = consensus.masked_fill(~allow, torch.inf).topk(min(KC, n), largest=False).indices
            at = torch.unique(torch.cat([*parts, best[allow[best]]]))
        else:
            at = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        cols, logs, ranks = [], [], []
        for key in SIGNALS:
            v = scores[key][at]
            finite = torch.isfinite(v)
            rank = (n - torch.searchsorted(ascending[key], v.contiguous(), right=True) + 1).float()
            lr = torch.where(finite, rank.clamp_min(1).log10(), torch.full_like(v, NO_RANK))
            mean, sd, top1, top10 = stats[key]
            nan = torch.full_like(v, float("nan"))
            cols += [torch.where(finite, v, nan), torch.where(finite, (v - mean) / sd, nan), lr,
                     torch.where(finite, v - top1, nan), torch.where(finite, v - top10, nan)]
            logs.append(lr)
            ranks.append(torch.where(finite, rank.clamp_min(1), torch.zeros_like(rank)))
        L = torch.stack(logs, 1)
        Ls = L.sort(dim=1).values
        cols += [(L <= 1).sum(1).float(), (L <= np.log10(50)).sum(1).float(), (L <= np.log10(200)).sum(1).float(),
                 (L <= 3).sum(1).float(), Ls[:, 0], Ls[:, 1], L.mean(1)]
        X = torch.stack(cols, 1).cpu().numpy()
        r = at.cpu().numpy()
        ranks = torch.stack(ranks, 1).cpu().numpy().astype(np.int64)
        bq, bc = np.float32(seed.bpm), self.bpm[r]
        with np.errstate(all="ignore"):
            d = np.abs(np.log2(bc / bq)) if np.isfinite(bq) and bq > 40 else np.full(len(r), np.nan, np.float32)
            d = np.where(bc > 40, d, np.float32(np.nan))
            fold = np.minimum(d, np.abs(d - 1))
        lq, lc = seed.code, self.codes_np[r]
        side = np.stack([fold, d, (lc == lq) & (lq >= 0), lc < 0, np.full(len(r), lq >= 0)], 1).astype(np.float32)
        in_graph = np.full((len(r), 1), seed.row is not None and bool(self.graph_np[seed.row]), np.float32)
        X = np.concatenate([X, side, self._scene_columns(seed, r, {} if scene_cache is None else scene_cache),
                            self.graph_np[r][:, None].astype(np.float32), in_graph], 1)
        # Only the songs inside the sound cap are scored: the others are in no body.
        capped = (ranks[:, 0] >= 1) & (ranks[:, 0] <= self.sound_cap)
        score = np.full(len(r), -np.inf)
        lr = X[capped, FEATURES.index("lr_snd")]
        score[capped] = self._predict(X[capped][:, self.columns]) - self.sound_weight * np.minimum(lr, 5.0)
        return Candidates(r, X, ranks, score)

    def _scene_columns(self, seed: Seed, rows: np.ndarray, cache: dict) -> np.ndarray:
        """graph, region, genre, aff, fwd, rev and nb of each row's artist against the seed's: NaN for a seed whose
        artist the scene table does not know. ``cache``: artist id -> its columns, for this seed's artist."""
        out = np.full((len(rows), 7), np.nan, np.float32)
        table, qa = self.scene, seed.artist_id
        if "profile" not in cache:
            related = table.related(qa) if table is not None and qa else None
            cache["profile"] = (table.profile(qa, related) if table is not None and qa else None, set(related or ()))
        profile, related = cache["profile"]
        if profile is None:
            return out
        artists, at = np.unique(self.scene_artist[rows], return_inverse=True)
        values = np.empty((len(artists), 7), np.float32)
        for i, a in enumerate(artists.tolist()):
            if a not in cache:
                stored = table.stored(a) if a else None
                g, rg, gn = components(profile, stored) if stored is not None else (np.nan,) * 3
                cache[a] = (g, rg, gn, affinity(profile, stored, a, self.scene_params) if a else 0.0,
                            float(a in related), float(qa in (table.related(a) or ())) if a else 0.0,
                            float(a in profile.neighbours))
            values[i] = cache[a]
        return values[at]

    def body(self, signals: Mapping[str, torch.Tensor], seed: Seed, keep: int = KEEP,
             scene_cache: dict | None = None) -> list[tuple[int, float, tuple[int, ...]]]:
        """The seed's body: (track id, body score, each signal's rank in ``SIGNALS`` order, 0 without a score) by
        falling score (ties: lower track id), the best song of each artist, at most ``keep``."""
        c = self.candidates(signals, seed, scene_cache=scene_cache)
        tids = self.ids[c.rows]
        out, seen = [], set()
        for j in np.lexsort((tids, -c.score)):
            if not np.isfinite(c.score[j]) or len(out) == keep:
                break
            artist = int(self.artist_np[c.rows[j]])
            if artist not in seen:
                seen.add(artist)
                out.append((int(tids[j]), float(c.score[j]), tuple(c.ranks[j].tolist())))
        return out

    def bodies(self, bases: Mapping[str, torch.Tensor], code: int, row: int | None = None,
               windows: np.ndarray | None = None, artist: str | None = None, artist_id: int | None = None,
               bpm: float | None = None) -> dict[str, list[tuple[int, float, tuple[int, ...]]]]:
        """The bodies of one seed: "same" among its language's songs (``code``: its language in ``codes``, -1 for
        none) and "all" among every language's, which reads the seed as one of unknown language. A stored seed is
        its sonic ``row``; a live one comes with its raw sonic ``windows``, credited ``artist``, scene-table
        ``artist_id`` and tempo."""
        signals = self.signals(bases, row, windows)
        cache: dict = {}
        out = {"all": self.body(signals, self.seed(-1, row, artist, artist_id, bpm), scene_cache=cache)}
        out["same"] = self.body(signals, self.seed(code, row, artist, artist_id, bpm), scene_cache=cache) \
            if code >= 0 else out["all"]
        return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="consensus", description="export a fitted consensus model for serving")
    p.add_argument("model", type=Path, help="pickle of {'model': HistGradientBoostingRegressor, 'feats': its "
                                            "feature names in column order}")
    p.add_argument("out", type=Path, help="npz of tree arrays (listener.Regressor)")
    p.add_argument("--meta", default=None, help="a note kept in the file (default: the pickle's name)")
    args = p.parse_args(argv)
    stored = pickle.loads(args.model.read_bytes())
    model, features = stored["model"], [str(f) for f in stored["feats"]]
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        p.error(f"the model reads features consensus.FEATURES lacks: {unknown}")
    exported = listener.export_regressor(model, features, args.meta or args.model.name)
    # The numpy evaluation must be scikit-learn's own, missing values included.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, len(features))).astype(np.float32) * 3
    X[rng.random(X.shape) < 0.2] = np.nan
    worst = float(np.abs(exported.predict(X) - model.predict(X)).max())
    if worst > 0:
        p.error(f"the exported model differs from scikit-learn's predict by {worst:g}")
    exported.save(args.out)
    print(f"{args.out}: {len(exported.roots)} trees, {len(exported.feature)} nodes, depth {exported.depth}, "
          f"{len(features)} features; max |difference| to scikit-learn on {len(X)} random rows with NaNs {worst:.3g}")


if __name__ == "__main__":
    sys.exit(main())
