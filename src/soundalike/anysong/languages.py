"""Sung-language labels for the any-song catalog and the lookups behind the same-language gate.

Each 30 s Deezer preview gets Whisper language identification (encoder plus one decoder step after
``<|startoftranscript|>``). The top-5 language probabilities and the no-speech probability are stored so
thresholds can change without relabelling:

    python -m soundalike.anysong.languages --ids index_full/melody/ids.npy \
        --previews C:/anysong/previews --out C:/anysong/languages \
        --model-dir C:/models/whisper-large-v3-turbo

writes resumable ``part-*.npz`` shards, then ``languages.npz`` (``ids``, ``vocab``, ``top_idx``,
``top_prob``, ``no_speech``, ``meta``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
EXCERPT_SECONDS = 30.0
TOP = 5
MIN_CONFIDENCE = 0.8
MIN_MARGIN = 0.5
# Best guess for a seed the gate cannot place (no label of its own, no artist majority): Whisper's top language when its
# probability is at least GUESS_MIN, unless the artist has labelled songs and under GUESS_VETO of them are in that
# language (so English songs by mostly Korean-singing acts stay unfiltered). On the catalog's unsure songs it covers
# 39 % and matches the lyrics' language 87 % of the time (docs/ANYSONG_EVAL.md, "A best-guess seed language",
# 2026-11-13).
GUESS_MIN = 0.6
GUESS_VETO = 0.5
UNKNOWN = "unknown"
MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


def load_excerpt(path: str | Path) -> np.ndarray:
    """First 30 s at 16 kHz, zero-padded like Whisper's own preprocessing."""
    import soxr

    from .audio import decode_mono

    mono, native_sr = decode_mono(path)
    x = soxr.resample(mono, native_sr, SAMPLE_RATE, quality="HQ") if native_sr != SAMPLE_RATE else mono
    n = int(SAMPLE_RATE * EXCERPT_SECONDS)
    return np.pad(x[:n], (0, max(0, n - len(x)))).astype(np.float32)


class WhisperLanguageID:
    """Batched Whisper language identification returning language and no-speech probabilities."""

    def __init__(self, model_dir: str | Path, device: str = "cuda") -> None:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(model_dir), local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa",
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        ).to(device).eval()
        items = sorted(self.model.generation_config.lang_to_id.items())
        self.vocab = tuple(token.removeprefix("<|").removesuffix("|>") for token, _ in items)
        self.lang_ids = torch.tensor([token_id for _, token_id in items], device=device)
        self.sot = int(self.model.generation_config.decoder_start_token_id)
        self.no_speech_id = int(self.processor.tokenizer.convert_tokens_to_ids("<|nospeech|>"))

    def __call__(self, waves: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        features = self.processor.feature_extractor(
            list(waves), sampling_rate=SAMPLE_RATE, return_tensors="pt", device=self.device,
        ).input_features.to(device=self.device, dtype=self.model.dtype)
        decoder_input_ids = torch.full((len(waves), 1), self.sot, device=self.device, dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(input_features=features, decoder_input_ids=decoder_input_ids,
                                use_cache=False).logits[:, -1].float()
            languages = torch.softmax(logits[:, self.lang_ids], dim=-1)
            no_speech = torch.softmax(logits, dim=-1)[:, self.no_speech_id]
        return languages.cpu().numpy(), no_speech.cpu().numpy()


def top_languages(probabilities: np.ndarray, top: int = TOP) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="stable")[:, :top]
    return order.astype(np.uint8), np.take_along_axis(probabilities, order, axis=1).astype(np.float32)


def decide(top_idx: np.ndarray, top_prob: np.ndarray, *, min_confidence: float = MIN_CONFIDENCE,
           min_margin: float = MIN_MARGIN) -> np.ndarray:
    """Vectorised ``v4_gates.decide_language``: vocab index, or -1 when not convincing."""
    if (
        top_idx.ndim != 2
        or top_prob.shape != top_idx.shape
        or top_prob.shape[1] < 2
        or not 0.0 < min_confidence <= 1.0
        or not 0.0 <= min_margin < 1.0
        or not np.isfinite(top_prob).all()
        or np.any((top_prob < 0.0) | (top_prob > 1.0))
    ):
        raise ValueError("language label probabilities, shapes, or thresholds are invalid")
    confident = (top_prob[:, 0] >= min_confidence) & (top_prob[:, 0] - top_prob[:, 1] >= min_margin)
    return np.where(confident, top_idx[:, 0].astype(np.int16), np.int16(-1))


def normalize_artist(name: str) -> str:
    folded = unicodedata.normalize("NFKD", str(name)).casefold()
    return " ".join("".join(c for c in folded if not unicodedata.combining(c)).split())


@dataclass(frozen=True)
class LanguageTable:
    """Decided language per indexed track (sorted ids), plus an artist-majority fallback and a best guess
    (``guess``) for tracks neither places."""

    ids: np.ndarray
    codes: np.ndarray
    vocab: tuple[str, ...]
    artists: dict[str, str]
    meta: dict
    top_codes: np.ndarray | None = None  # Whisper's top language per track, aligned with ``ids``
    top_probs: np.ndarray | None = None
    # Artists with labelled tracks but no majority: the languages of at least ``guess_veto`` of those tracks.
    spread: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, *, min_confidence: float = MIN_CONFIDENCE, min_margin: float = MIN_MARGIN,
             artist_names: dict[int, str] | None = None, min_artist_tracks: int = 2,
             min_artist_share: float = 0.8, guess_veto: float = GUESS_VETO) -> "LanguageTable":
        with np.load(path, allow_pickle=False) as z:
            ids = z["ids"].astype(np.int64)
            vocab = tuple(str(v) for v in z["vocab"])
            top_idx = z["top_idx"]
            top_prob = z["top_prob"]
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
        if (
            not vocab
            or len(set(vocab)) != len(vocab)
            or top_idx.ndim != 2
            or top_idx.shape[0] != len(ids)
            or top_prob.shape != top_idx.shape
            or top_idx.shape[1] < 2
            or not np.issubdtype(top_idx.dtype, np.integer)
            or (top_idx.size and (int(top_idx.min()) < 0 or int(top_idx.max()) >= len(vocab)))
            or (top_prob.size and np.any(np.diff(top_prob, axis=1) > 1e-6))
            or not isinstance(meta, dict)
            or (len(ids) and np.any(ids <= 0))
        ):
            raise ValueError(f"invalid language table structure in {path}")
        codes = decide(top_idx, top_prob, min_confidence=min_confidence, min_margin=min_margin)
        order = np.argsort(ids, kind="stable")
        ids, codes = ids[order], codes[order]
        if len(ids) and np.any(ids[1:] == ids[:-1]):
            raise ValueError(f"duplicate track ids in {path}")
        table = cls(ids, codes, vocab, {}, meta, top_idx[order, 0].astype(np.int16),
                    top_prob[order, 0].astype(np.float32))
        if artist_names:
            votes = table.artist_votes(artist_names)
            table.artists.update(table._majority(votes, min_tracks=min_artist_tracks, min_share=min_artist_share))
            for artist, counter in votes.items():
                if artist and artist not in table.artists:
                    total = sum(counter.values())
                    table.spread[artist] = tuple(sorted(table.vocab[code] for code, n in counter.items()
                                                        if n / total >= guess_veto))
        return table

    def __len__(self) -> int:
        return len(self.ids)

    def codes_for(self, track_ids: Iterable[int]) -> np.ndarray:
        query = np.asarray(list(track_ids), dtype=np.int64)
        if not len(self.ids) or not len(query):
            return np.full(len(query), -1, dtype=np.int16)
        pos = np.clip(np.searchsorted(self.ids, query), 0, len(self.ids) - 1)
        return np.where(self.ids[pos] == query, self.codes[pos], np.int16(-1))

    def languages(self, track_ids: Iterable[int]) -> list[str]:
        return [self.vocab[c] if c >= 0 else UNKNOWN for c in self.codes_for(track_ids).tolist()]

    def language(self, track_id: int) -> str:
        return self.languages([track_id])[0]

    def labelled(self, track_id: int) -> bool:
        return bool(len(self.ids)) and bool(self.ids[min(np.searchsorted(self.ids, track_id),
                                                             len(self.ids) - 1)] == track_id)

    def artist_language(self, artist: str | None) -> str:
        return self.artists.get(normalize_artist(artist), UNKNOWN) if artist else UNKNOWN

    def effective_codes(self, track_ids: np.ndarray, artist_of: Callable[[int], str | None] | None = None
                        ) -> np.ndarray:
        """Vocab index (-1: unknown) of each track's language as the same-language gate reads it, for many tracks at
        once: the track's own label, else (with ``artist_of``, track id -> credited artist) its artist's majority
        language (``artist_language``)."""
        ids = np.asarray(track_ids, dtype=np.int64)
        codes = self.codes_for(ids).astype(np.int16)
        if artist_of is None or not self.artists:
            return codes
        code_of = {language: code for code, language in enumerate(self.vocab)}
        by_name: dict[str, int] = {}
        unsure = np.flatnonzero(codes < 0)
        for i, tid in zip(unsure.tolist(), ids[unsure].tolist()):
            name = artist_of(tid)
            if not name:
                continue
            code = by_name.get(name)
            if code is None:
                code = by_name[name] = code_of.get(self.artist_language(name), -1)
            codes[i] = code
        return codes

    def guess(self, track_id: int, artist: str | None, *, min_prob: float = GUESS_MIN) -> str:
        """A best guess for a track without a label of its own: Whisper's top language when its probability is at
        least ``min_prob`` and the artist's labelled tracks do not rule it out (``spread``); "unknown" otherwise."""
        if self.top_codes is None or self.top_probs is None or not len(self.ids):
            return UNKNOWN
        pos = min(int(np.searchsorted(self.ids, track_id)), len(self.ids) - 1)
        if self.ids[pos] != track_id or self.top_probs[pos] < min_prob:
            return UNKNOWN
        language = self.vocab[int(self.top_codes[pos])]
        allowed = self.spread.get(normalize_artist(artist)) if artist else None
        return UNKNOWN if allowed is not None and language not in allowed else language

    def artist_votes(self, artist_names: dict[int, str]) -> dict[str, Counter]:
        """Artist -> labelled tracks per language code."""
        tids = list(artist_names)
        votes: dict[str, Counter] = {}
        for tid, code in zip(tids, self.codes_for(tids).tolist()):
            if code >= 0:
                votes.setdefault(normalize_artist(artist_names[tid]), Counter())[code] += 1
        return votes

    def artist_majority(self, artist_names: dict[int, str], *, min_tracks: int, min_share: float
                        ) -> dict[str, str]:
        """Artist -> language when >= ``min_share`` of their confidently labelled tracks agree."""
        return self._majority(self.artist_votes(artist_names), min_tracks=min_tracks, min_share=min_share)

    def _majority(self, votes: dict[str, Counter], *, min_tracks: int, min_share: float) -> dict[str, str]:
        out = {}
        for artist, counter in votes.items():
            code, count = counter.most_common(1)[0]
            if artist and count >= min_tracks and count >= min_share * sum(counter.values()):
                out[artist] = self.vocab[code]
        return out


class LanguageRows:
    """The gate's language of every row of each channel index (``LanguageTable.effective_codes``), held on the
    index's device: boolean row masks for searches among one language's songs."""

    def __init__(self, table: LanguageTable, indexes: Mapping[str, object],
                 artist_of: Callable[[int], str | None] | None = None):
        import torch

        self.vocab = {language: code for code, language in enumerate(table.vocab)}
        self.codes: dict[str, object] = {}
        self.counts: dict[str, np.ndarray] = {}
        done: list[tuple[np.ndarray, np.ndarray]] = []  # channels usually share one id order
        for name, ix in indexes.items():
            ids = np.asarray(ix.ids, dtype=np.int64)
            codes = next((c for other, c in done if np.array_equal(other, ids)), None)
            if codes is None:
                codes = table.effective_codes(ids, artist_of)
                done.append((ids, codes))
            self.codes[name] = torch.as_tensor(codes, device=ix.id_tensor.device)
            self.counts[name] = np.bincount(codes.astype(np.int64) + 1, minlength=len(table.vocab) + 1)

    def masks(self, language: str) -> dict[str, object] | None:
        """Channel -> boolean row mask of the songs in ``language`` or of unknown language (what the gate keeps);
        None for a language outside the table's vocabulary ("unknown" among them)."""
        code = self.vocab.get(language)
        if code is None:
            return None
        return {name: (codes == code) | (codes < 0) for name, codes in self.codes.items()}

    def pool(self, language: str) -> int:
        """How many songs of the sonic index (else of the first channel's) pass ``masks(language)``."""
        counts = self.counts["sonic"] if "sonic" in self.counts else next(iter(self.counts.values()))
        return int(counts[0] + counts[self.vocab[language] + 1])



def merge_shards(out: Path, vocab: Sequence[str], meta: dict) -> Path:
    from .embed_run import load_store

    store = load_store(out)
    if not store:
        raise ValueError(f"no shards in {out}")
    order = np.argsort(store["ids"], kind="stable")
    target = out / "languages.npz"
    temporary = target.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, ids=store["ids"][order].astype(np.int64), vocab=np.asarray(vocab),
                top_idx=store["top_idx"][order], top_prob=store["top_prob"][order],
                no_speech=store["no_speech"][order], meta=np.asarray(json.dumps(meta, sort_keys=True)))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_ids(paths: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for p in paths:
        ids += np.load(p).astype(np.int64).tolist() if p.endswith(".npy") else [int(x) for x in Path(p).read_text().split()]
    return list(dict.fromkeys(ids))


def main(argv: list[str] | None = None) -> None:
    from .embed_run import decoded_batches, done_ids, preview_path

    p = argparse.ArgumentParser(prog="languages")
    p.add_argument("--ids", required=True, nargs="+", help=".npy or whitespace-separated .txt track id lists")
    p.add_argument("--previews", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir", required=True, help=f"local snapshot of {MODEL_ID}@{MODEL_REVISION[:12]}")
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--shard", type=int, default=4800)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda", help="torch device; cpu runs the model in float32")
    p.add_argument("--fp16-accumulation", action="store_true",
                   help="~1.2x faster on RTX 40/50; changed 1 of 480 decisions (unknown->en) in testing")
    args = p.parse_args(argv)
    if min(args.batch, args.shard, args.workers) < 1 or args.limit < 0:
        p.error("batch, shard and workers must be positive; limit must be non-negative")

    root, out = Path(args.previews), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = read_ids(args.ids)
    have = done_ids(out)
    todo = [t for t in ids if t not in have and preview_path(root, t).exists()]
    missing = sum(1 for t in ids if not preview_path(root, t).exists())
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(ids)} ids, {len(have)} done, {missing} without preview, {len(todo)} to label", flush=True)

    lid = WhisperLanguageID(args.model_dir, args.device)
    if args.fp16_accumulation:
        lid.torch.backends.cuda.matmul.allow_fp16_accumulation = True
    meta = {"model": MODEL_ID, "revision": MODEL_REVISION, "sample_rate": SAMPLE_RATE,
            "excerpt_seconds": EXCERPT_SECONDS, "top": TOP, "fp16_accumulation": bool(args.fp16_accumulation)}
    (out / "vocab.json").write_text(json.dumps({"vocab": lid.vocab, **meta}, indent=1))

    failures: list[int] = []

    def decode(tid: int):
        try:
            return tid, load_excerpt(preview_path(root, tid))
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"decode failed {tid}: {exc}", flush=True)
            return tid, None

    shard_idx = 1 + max((int(s.stem.split("-")[1]) for s in out.glob("part-*.npz")), default=-1)
    buf: dict[str, list] = {"ids": [], "top_idx": [], "top_prob": [], "no_speech": []}
    t0, n_done = time.time(), 0

    def flush() -> None:
        nonlocal shard_idx
        if not buf["ids"]:
            return
        path = out / f"part-{shard_idx:04d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, ids=np.asarray(buf["ids"], dtype=np.int64),
                         **{k: np.concatenate(buf[k]) for k in ("top_idx", "top_prob", "no_speech")})
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        shard_idx += 1
        for v in buf.values():
            v.clear()

    with ThreadPoolExecutor(args.workers) as ex:
        for decoded in decoded_batches(ex, decode, todo, args.batch):
            ok = [(tid, clip) for tid, clip in decoded if clip is not None]
            failures += [tid for tid, clip in decoded if clip is None]
            if not ok:
                continue
            probs, no_speech = lid([clip for _, clip in ok])
            if not (np.isfinite(probs).all() and np.isfinite(no_speech).all()):
                raise ValueError("non-finite language probabilities")
            idx, prob = top_languages(probs)
            buf["ids"].extend(tid for tid, _ in ok)
            buf["top_idx"].append(idx)
            buf["top_prob"].append(prob)
            buf["no_speech"].append(no_speech.astype(np.float32))
            n_done += len(ok)
            if len(buf["ids"]) >= args.shard:
                flush()
                rate = n_done / (time.time() - t0)
                print(f"{n_done}/{len(todo)} {rate:.1f} tracks/s eta {(len(todo) - n_done) / rate / 60:.1f} min",
                      flush=True)
    flush()
    failure_path = out / "failures.txt"
    if failures:
        failure_path.write_text("\n".join(map(str, failures)))
        raise RuntimeError(f"{len(failures)} previews failed decoding; see {failure_path}; repair previews and resume")
    failure_path.unlink(missing_ok=True)
    meta.update(labelled=len(done_ids(out)), requested=len(ids), without_preview=missing,
                decode_failures=0, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    target = merge_shards(out, lid.vocab, meta)
    print(f"done {n_done} in {(time.time() - t0) / 60:.1f} min; {len(failures)} decode failures; wrote {target}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
