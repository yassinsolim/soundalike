# Inspector Feedback — Iteration 13

## Verdict: FAIL (goal) · Method/protocol integrity: PASS · Commit hygiene: FAIL

Independent inspection of commit `52c483e` ("feat(ml): [B] build compact CLAP
catalog index") against the immutable `goal.md` and `inspector-feedback-12.md`.
Nothing the Builder claimed was trusted. I recomputed every JSON `content_sha256`,
re-verified both Ed25519 signatures with `ssh-keygen`, re-checked the trust anchors
pinned in the importer against the committed bytes, recomputed the served-lists and
semantic-order hashes, re-hashed the local (gitignored) compact and full CLAP assets,
reconciled the coverage arithmetic, ran the full test suite, scanned the diff for
secrets, confirmed the exact changed-file set, and **reconstructed the actual top-5
lists for 20 diverse seeds** (mapping opaque IDs back through the catalog index) to
judge scene coherence.

**One sentence:** This is a clean, honest, fully-signed development iteration that
builds a real 272,709-row preview-derived CLAP index, freezes a proxy-safe blinded
human A/B pack, ships nothing to production, and correctly refuses to claim AC#3 —
but the goal is still **FAIL** because the deciding human evidence (AC#3) does not
yet exist, and the commit again carries the **prohibited `Assisted-by:` attribution
trailer** (a third-time regression).

---

## What I independently reproduced (nothing trusted)

**Preregistration chain — VERIFIED.** All three revisions self-hash correctly and
the supersession chain is coherent: r1 `30262e24…` → r2 supersedes r1 (`c34d76da…`)
→ r3 supersedes r2 (`2c1bb55c…`). The module pins `PREREGISTRATION_SHA256` = the r3
hash. r1 was a bounded-throughput preflight (no artifact, cache discarded); r2
completed the full extraction but failed the strict geometry gate; r3 relaxed only
the catalog-geometry retention thresholds. All transitions are documented in the
`preflight_disposition` fields.

**Ed25519 signatures — VERIFIED GOOD.**
- `preregistration-v13-r3.json` → `ssh-keygen -Y verify` **Good "soundalike-clap-prereg"** signature.
- `state.json` → **Good "soundalike-human-eval-v13"** signature.
- The importer's pinned trust anchors (`TRUSTED_PREREGISTRATION_R3_FILES`,
  `TRUSTED_V13_FILES`, `TRUSTED_V13_{PROTOCOL,LISTS,STATE}`) all match the committed
  file bytes/content hashes exactly. The served evaluator HTML hash
  (`d524d8f7…`) equals the `evaluator_sha256` bound into the signed protocol/state.
- **No private key material committed** — only `.pub`, `allowed_signers`, and `.sig`
  files exist in both v13 directories.

**Cross-artifact hash binding — VERIFIED.** protocol.served_lists_sha256 =
state.served_lists_sha256 = `content_hash(served-lists)` = `8c09b31e…`;
semantic_order_hash(served-lists) = `405a56b6…` matches protocol/state; state
protocol_sha256 = protocol content hash; state compact_asset_sha256 = geometry
asset sha; state diagnostics hash = variant-diagnostics content hash;
signature-metadata state_sha256 = file hash of state.json. Every link closes.

**Catalog identity / 272,853 alignment — VERIFIED.** `track_ids_tobytes_sha256`
(`a20632fc…`) and `source_index_sha256` (`f3ed57af…`) are identical across prereg,
coverage, diagnostics, and the untouched production `deepvibe_index_v5.npz`. The
module fails closed on any row-count, uniqueness, or row-order drift.

**Coverage / status integrity — VERIFIED.** available **272,709** + no_preview
**144** + pending **0** + error **0** = **272,853** (terminal_fraction 1.0,
available_fraction 99.947%). 303 transient `OSError` attempts recovered; detail
retained in gitignored SQLite.

**Streaming / no retained audio — VERIFIED (as reported).** `retained_audio_files
0`, `temporary_files_remaining 0`, `signed_preview_urls_persisted false`; only the
aligned float16 memmap, SQLite status, and derived compact assets remain in
gitignored `ml_data/clap_v13`. (Confirmed those files exist locally and are not
tracked by git.)

**Runtime / GPU — plausible and within budget.** 27,738 s (7 h 42 m) at 9.83
tracks/s, 130.6 GB streamed, peak CUDA **3.95 GB allocated / 6.09 GB reserved** —
fits the RTX 5080's 16 GB comfortably.

**JL128 index — label-independent, VERIFIED.** Gaussian orthogonal JL, seed
`20260713`, `fit_labels: none`, selection on an independent unlabeled 256×20,000
catalog sample. Only **dim 128 passes** the (r3) geometry gates
(pair-cosine Spearman 0.9524, mean top-50 overlap 0.7563, p05 0.58, union-rank
Spearman 0.588); 64/96/112 each fail ≥1 gate. Float16 reload preserves the metrics
(0.9523614 vs 0.9523613). Asset **69,850,496 B (66.6 MB) < 70 MB**, sha
`0c204bd…`. I re-hashed the local `compact-clap128.f16.npy` and `full-clap512.f16.npy`:
**byte counts and SHA-256 match the artifacts exactly** — the evidence is real, not
fabricated.

**Variant selection — predeclared, no commercial-label tuning — VERIFIED.**
`commercial_human_ratings_used: 0`, `proxy_evidence_is_deciding: false`,
`old_gnod_co_primary_used: false`. Fixed order conservative→graph→pure; the first to
pass every proxy gate is frozen. Gate results (all 60 seeds, 300 slots):

| variant | passes | junk | same-artist | max track/artist slot | uniq-artist | style Δ vs prod | Deezer-affinity Δ |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **conservative_clap_fallback** (selected) | ✅ | 0 | 0 | 0.67% / 1.33% | 91.7% | −0.003 | **+0.224** |
| graph_clap_union | ✅ | 0 | 0 | 0.33% / 1.0% | 91.7% | −0.026 | +0.20 |
| pure_clap | ❌ | 0 | 0 | 0.33% / 1.0% | 95.3% | −0.042 | **−0.095** |

`pure_clap` is honestly **rejected** for regressing the Deezer related-artist proxy
(−0.095 < −0.05 gate). Junk/same-artist/hubness/concentration diagnostics are clean
for the selected variant.

**v13 blinded pack — VERIFIED.** 60 seeds, **13** distinct scenes (rap 2, rnb 8,
indie 8, shoegaze 2, hyperpop 5, electronic 5, metal 3, jazz 3, city/J/K-pop 6,
latin/afrobeats 3, difficult_blend 3, pop 6, rock 6), 2 opaque lists × 5 per seed,
per-session randomized order, opaque salted `S13-/L13-/T13-` IDs (blinding salt is
`secrets.token_hex(32)` at build time — **not committed**), `RANKINGS_LOCKED`,
`ratings_count_at_freeze 0`.

**Fresh preview coverage — VERIFIED.** Ranked positions **600/600** resolvable
(fraction 1.0), 478/478 unique results, **59/60** seeds (the lone `no_preview` seed
`646715112` is the same one carried since v11 and is handled), observed
`Cache-Control: no-store`, no persisted signed URLs. `chrome_playback` honestly
"pending manual verification."

**Evaluator compatibility / privacy — VERIFIED.** The `human_eval_v11` change is a
backward-compatible optional `resolver` injection seam (default preserved). The
`human_aggregate_v10` change is additive: it accepts schema 13 alongside 10 with a
**stronger** binding (`_verify_v13_state`: content-hash pins, RANKINGS_LOCKED,
ratings 0, served/semantic binding, ssh-verified state signature) and keeps the ≥3
independent collector-signed-rater gate. The evaluator HTML enforces CSP
`default-src 'none'` (`connect-src 'self' https://soundalike.yassin.app`), fetches
only `/api/preview?id=<numeric>` / `/protocol.json` / `/served-lists.json`, sends
**only the numeric Deezer ID**, and **exports ratings as a local file — never
uploads them**; localStorage is local-only.

**Prospective resources / parity — VERIFIED, honest.** Production index unchanged
(`f3ed57af…`). Compact index adds only ~49 MB touched RSS; compact query latency
mean 110 ms vs production 334 ms. Vercel project tier "unknown", `fit_claimed:
false` — no silent fit claim. Deployment explicitly prohibited this iteration
(`deployed: false`, `release_asset_uploaded: false`, `wired: false`).

**Quality gate — PASS (reproduced).** `.\.venv\Scripts\python.exe -m pytest tests\
-q` → **534 passed in 34.5 s** (matches the claimed 534; v12 was 526). Build and
`pip_audit` (0 vulns, Pillow ≥12.3.0 remediation) are Builder-reported and were
**not** independently re-run here (long build / network), but the test gate — the
declared quality command — reproduces.

**Security — CLEAN.** No credential/private-key findings in the diff; all
`secret/password/token` matches are benign docs, the runtime blinding-salt call, or
`*_sha256` field names.

**Production unchanged — VERIFIED.** Changed set is the new/edited ML+eval modules,
their test, README, CASE_STUDY, pyproject, `.gitattributes`, the blinded rater HTML,
and the v13 `.goals` protocol/artifacts. **No** webapp, API, production recommender,
`deepvibe_index_v5.npz`, index manifest, or deployment asset is touched. Every
report flags `production_changed/deployed/ac3_claimed/commercial_final_opened =
false`; `production_version_retained: 2026.07.11-dual-sonic64`.

---

## Direct list inspection — 20 diverse seeds (not treated as human gold)

I reconstructed real titles/artists for 20 seeds spanning all 13 scenes. Coherence
is broadly strong (impeccable bebop for the Dizzy Gillespie jazz seed; textbook
metal for Avenged Sevenfold; correct K-pop for TWICE; correct reggaeton for Bad
Bunny; strong indie-revival for The Strokes/Killers baselines).

**One obvious cross-scene failure, and it is in the CHALLENGER, not the baseline:**
For the shoegaze seed **"You Wind Me Up — Nothing" (DEV-SONIC-037)**, the production
**baseline** correctly returns shoegaze/alt neighbors (Sunny Day Real Estate,
Deftones, Catherine Wheel…), but the selected **conservative CLAP challenger**
(gate_fired, `lastfm_confidence 0.985`) returns **drum & bass / dubstep** — ASC,
Source Direct, Submorphics, Cookie Monsta. This is a high-confidence **Last.fm
artist-name collision** on the extremely ambiguous name "Nothing"; on this seed the
challenger is clearly **worse** than baseline, yet it passed the *aggregate* proxy
gates (style/junk/slot metrics averaged over 60 seeds mask a single-seed collapse).
Per the task, I flag this as a candidate failure for **human raters to adjudicate**,
not as a verdict. It is not a blocker (nothing ships; the blinded A/B is precisely
the instrument meant to catch it), but it is concrete evidence the challenger can
regress per-seed and that an artist-disambiguation guard would help.

**Supporting caution:** the `candidate_recall_diagnostic` is very low
(recall@50 ≈ 0.028 ≈ 1/36 known category-A targets, for all variants). Combined with
`pure_clap`'s Deezer-affinity regression and the fact the selected challenger is
Last.fm-dominant with **24/60 seeds falling back to exact production**, the actual
CLAP contribution reaching the human pack is minor. That is an honest, defensible
conservative choice — but raters are largely comparing production against a
Last.fm-flavored production-adjacent list on 40% of seeds.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / actual lists.** MET; v10/v11 packs remain
  byte-identical, v13 adds a separate 60-seed / 13-scene pack recording real lists.
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET and extended;
  three predeclared CLAP variants with an honest rejection of `pure_clap`.
- [ ] **AC#3 — ≥20% gain / no scene <−10% / ≥80% coherent top-5 / no junk.**
  **NOT MET (dispositive).** Zero human commercial-list ratings; no `sonic_human`
  report; correctly not claimed. Sole substantive blocker.
- [x] **AC#4 — External validation, not same-artist-only.** MET; MusicBrainz /
  Deezer-affinity / Last.fm used as proxy safety, explicitly non-deciding.
- [~] **AC#5 — Wired into desktop + hosted, live-verified.** N/A this iteration
  (deployment prohibited; nothing wired). Carried: preview hardening still staged.
- [x] **AC#6 — Resources measured, no silent fallback.** MET; runtime/GPU/index/
  latency/RSS recorded; Vercel fit explicitly not claimed.
- [x] **AC#7 — Regression tests + full suite + docs.** MET; 534 green; README and
  CASE_STUDY honestly document the hypothesis, MTAT-overlap caveat, method,
  proxy-only role, and zero-ratings/no-AC3-claim status.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — independently reproduced **534 passed** (~34.5 s).
- Note: the gate confirms code/report correctness only; it does not adjudicate AC#3.

---

## Issues Found (ranked)

1. **[BLOCKER — expected] AC#3 has no human evidence.** `ratings_count == 0`; no
   `sonic_human` report. Correctly not claimed. The only substantive reason the goal
   is not yet achievable.
2. **[COMMIT-HYGIENE VIOLATION — must fix, third regression] Prohibited attribution
   trailer.** Commit `52c483e` body ends with `Assisted-by: Claude:Sonnet-4.6`.
   `goal.md`: "Do not add attribution trailers or generated-by taglines." Flagged at
   iterations 9 and 12; still present. Drop it on all future `[B]` commits.
3. **[QUALITY — for raters] Challenger scene collapse on ambiguous artist name.**
   Shoegaze "Nothing" → drum&bass/dubstep via a 0.985-confidence Last.fm name
   collision; passed aggregate proxy gates. Add a per-seed sanity/disambiguation
   guard for ambiguous names, or expect human raters to penalise it.
4. **[RIGOR — transparent caveat] Post-hoc geometry-gate relaxation.** r2's strict
   gate (Spearman ≥0.99) failed every dimension; r3 relaxed to ≥0.94 (and top-50
   0.75→0.74, union-rank 0.6→0.55) after observing r2 geometry. This is a
   compression-fidelity threshold, not a human/MTAT-label metric, and is disclosed —
   acceptable for development, but note the thresholds were tuned to observed data.
5. **[TRACEABILITY — minor] r3 build predates its nominal freeze.** Coverage
   `first_completed_at 2026-07-13T07:26:07` precedes r3 `frozen_at 07:30:00` by ~4
   min, contradicting "starts … after this file is signed." CLAP extraction is
   deterministic and label-independent, so there is no leakage, but the `frozen_at`
   (a round-minute nominal value) should be reconciled with the actual signing time.

## Credit where due
Every signature, hash, binding, and asset I checked is internally consistent and
independently reproducible; the local CLAP assets match their recorded SHAs to the
byte; `pure_clap` is honestly rejected; the MTAT pretraining-overlap caveat from
iteration 12 is now stated in the README; and nothing was shipped or claimed. The
blinding, trust-anchor pinning, and schema-13 aggregator hardening are exemplary.

---

## Evaluator readiness & minimum next action

**READY TO RATE.** The pack is frozen, hash-bound, state-signed, `RANKINGS_LOCKED`,
ratings 0, preview coverage 600/600 positions, evaluator privacy-tight, and the
aggregator now verifies and consumes schema-13 studies behind a ≥3-independent-rater
gate.

**Minimum next action to move toward PASS:**
1. Recruit ≥3 (ideally 5) independent raters; run the frozen v13 blinded A/B; collect
   collector-Ed25519-signed exports; aggregate into a `sonic_human` report.
2. Test AC#3 on that human evidence with margin (≥20% primary-score gain, ≥80%
   coherent top-5, CI excluding zero, no scene regressing >10%, no junk/version) —
   and expect the "Nothing"-type collision to surface; fix it before any promotion.
3. Drop the `Assisted-by:` trailer on future `[B]` commits.

## What Must Be Fixed (to reach PASS)
- Collect and aggregate **real** signed human ratings and satisfy **AC#3** on them;
  until then the goal remains **FAIL** by design.
