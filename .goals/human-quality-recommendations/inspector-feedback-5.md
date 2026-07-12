# Inspector Feedback — Iteration 5

## Verdict: FAIL

Independent inspection of commit `5f5973f` ("feat(recs): [B] add
collaborative candidate retrieval") against the immutable `goal.md` and
`inspector-feedback-4.md`. Every Builder claim was treated as untrusted and
re-derived from the frozen `protocol-v6` artifacts, the production index, the
committed rankings, the test suite, and independent web verification of the
dataset provenance.

**Bottom line:** This is a genuine, well-scoped, honestly-reported iteration
that does exactly what iteration 4 recommended — it builds an *independent,
query-conditioned collaborative candidate source* and measures candidate
recall as a gating metric. It also **fixes all three iteration-4 integrity
issues**. But the single **deciding** criterion **AC#3 fails on BOTH of its
sub-gates**, and the Builder correctly reports this itself (`final_pass=false`,
`passes_gate=false`, `deployed=false`). The deciding, leakage-masked method is
not merely flat — it is **~97% *worse* than the frozen baseline** on the
once-opened FINAL. The immutable goal is therefore not satisfied.

---

## What I reproduced (verified independently)

**Freeze integrity (real, tamper-evident, and now correctly gated).**
- Recomputed SHA-256 of the frozen benchmark
  (`ef6a861…`), index (`f3ed57a…`), and deciding masked asset (`29d1654…`) —
  **all match `protocol-v6/state.json` exactly**.
- **Both detached Ed25519 seals verify** via `ssh-keygen -Y verify`
  (namespace `soundalike-protocol`): `Good` for both `state.sig` and
  `frozen-state.sig`.
- `final_open_count=1` (opened once), `rankings_locked_before_open=true`,
  `locked_at=2026-07-12T11:57:27Z` recorded **before** the FINAL open at
  `11:59:29Z`. The iteration-4 `RANKINGS_LOCKED` bypass is **fixed**.
- DEV report `final_labels_compared=false`; DEV/FINAL artist-disjoint
  (`artist_overlap=[]`, 414 artists, 100 scenes, 107 dev / 88 final pairs).

**Dataset provenance (independently web-verified — real, not fabricated).**
Zenodo record `10.5281/zenodo.6609677` confirms Music4All-Onion: 109,269
pieces, **252,984,396 listening records of 119,140 users** (Last.fm), Moscati
et al., CIKM 2022, and `license: cc-by-4.0`. These match the Builder's
`collaborative-source-audit-v6.json` field-for-field. The corpus claims are
consistent: **22,849,039 masked tokens**, **115,468 users with ≥2 mapped
tracks** (of 119,140), two real item2vec models trained (~349 s each).

**Deciding FINAL (88 pairs, once-opened, `open_number=1`)** — recomputed from
`collaborative-final-once-v6.json`:

| FINAL method | primary | R@5 | R@50 | MRR | vs baseline |
|---|---|---|---|---|---|
| production_baseline | 0.008072 | 0.0114 | 0.0114 | 0.00568 | — |
| iteration3_deployed (live) | 0.011563 | 0.0114 | 0.0341 | 0.01196 | +43% |
| **winner (edge-masked hybrid)** | **0.000232** | **0.000** | 0.0227 | 0.00070 | **−97.1%** |
| collaborative_only (masked) | 0.000220 | 0.000 | 0.0227 | 0.00066 | −97.3% |
| audio_only | 0.000000 | 0.000 | 0.000 | 0.000 | −100% |
| hybrid_unmasked_ablation (LEAKY, invalid) | 0.013924 | 0.0114 | 0.0568 | 0.00859 | +72% |

- The deciding, leakage-masked **winner regresses −97.1%** vs the frozen
  baseline: `absolute_delta=−0.00784`, `ci95=[−0.0242, 0.0005]`,
  `probability_positive=0.314`, improved **2/88**, worsened 1, unchanged 85.
- **All eight pass gates are `false`** (`relative_gain`, `positive_absolute`,
  `minimum_absolute_gain`, `recall_at_10_non_regression`, `mrr_non_regression`,
  `ci_excludes_zero`, `meaningful_count`, `scene_no_regression`).
  **`final_pass=false`.**
- **Scene regression breach:** `punk-hardcore` collapses `0.1184 → 0.0`
  (**−100%**), far past the goal's −10% cap.
- The **only** method that beats baseline is the **unmasked ablation**
  (+72%), which the Builder itself flags `valid_deciding_method=false`: it
  memorizes the direct co-listening edges the benchmark measures. Its CI
  `[−0.0183, 0.0307]` still includes zero and its MRR regresses. The
  masked→unmasked gap is honest evidence that the apparent lift **is edge
  memorization, not topology**.

**Candidate generation did improve (the iter-4 ask), but not enough.**
FINAL candidate-recall@1000 rose from **audio 0.0114 → collab-masked 0.0455 →
hybrid-masked 0.0909**. Real, but **~91% of FINAL targets never enter even a
1,500-wide union**, so the reranker is still ordering a near-empty pool.

**DEV improved but did not generalize (and was not robust even on DEV).**
DEV hybrid primary 0.0190 vs production 0.0063 (`relative_gain=+201%`), but
`ci95=[−0.0099, 0.0375]` **includes zero**. The DEV-selected scorer is
`collaborative_cosine=0.5 + collaborative_reciprocal_rank=0.5 +
audio_blend=0.01` (all else 0) — i.e. **effectively collaborative-only; audio
is discarded** — and it collapsed on FINAL, the same DEV→FINAL overfit
signature as iteration 4.

**Direct human-list gate ALSO fails.** `collaborative-direct-judgments-v6.json`:
**13/20** difficult seeds pass; required **16**; `passes_gate=false`. Of the 7
failures: **4 are cold-start** (glaive ×2, Anri, Coltrane — *absent from the
272k catalogue*) and **3 are genuine scene-coherence misses** (Pixies → trip-hop
in positions 1–2; Post Malone → rap-only; JVKE → shoegaze/indie for a
piano-pop seed, position 1 unrelated).

---

## Root-cause decomposition (miss ranks & map coverage)

The near-zero FINAL recall is **not one problem** — I separated the causes:

1. **Sparse catalogue mapping (dominant, structural).** Only **13,680 /
   272,853 = 5.0%** of the served catalogue has a collaborative vector
   (Music4All→catalogue track mapping rate 26%). Consequently only **39/88
   (44%) FINAL pairs have both items mapped**; the collaborative signal is
   *structurally incapable* of helping the other 56% of pairs. This is the #1
   ceiling.
2. **Cold-start / catalogue coverage.** 4/20 held-out difficult seeds are not
   in the catalogue at all → automatic direct-list failures independent of any
   model or ranking.
3. **Join sparsity, not index absence (title/version).** Iteration 4 already
   proved every FINAL query+target exact-matches in the index; the *new*
   coverage loss is the Music4All→catalogue **join** (5% coverage), not song
   availability.
4. **Graph geometry / source mismatch.** Even for mapped pairs, Last.fm
   listening-order item2vec neighbours ≠ ListenBrainz session neighbours
   (masked collaborative-only FINAL recall@1000 = 4.5%). Two different
   behavioural sources produce different neighbourhoods.
5. **One-positive benchmark noise (measurement defect).** 88 seeds × **exactly
   one** gold counterpart in a 272k catalogue. Every method's R@5 ∈ {0, 1/88},
   baseline primary ≈ 0.008 sits on the floor, and every CI includes zero.
   The metric has almost no dynamic range, so it can only ever move by
   single-pair flips at the noise floor.

**Is single-counterpart pair retrieval a meaningful proxy for top-5 user
satisfaction? No.** With one designated counterpart among 272k tracks, a
subjectively excellent recommender that returns five perfect soundalikes will
usually still miss *the one* gold target (there are dozens–hundreds of equally
valid soundalikes; the benchmark rewards only one). This is exactly why the two
AC#3 axes disagree here — pair-retrieval R@5 ≈ 0 while direct list inspection
passes 13/20. The retrieval metric, **as constructed**, cannot demonstrate a
"clear" ≥20% human-aligned gain; it is a high-variance, low-ceiling proxy.

**Important caveat on the recommendation:** the fix is **not** to swap in an
easier metric. The goal already contains the harder, satisfaction-aligned gate
— "≥80% of a held-out set of 20 difficult seeds have a coherent top-5" — and
**it fails at 13/20 (65%)**. The right move is to make the primary score
higher-dynamic-range *without lowering the deciding bar* (see below), and to
actually clear the 80% coherence gate.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / held-out / credible sources.** MET.
  `protocol-v6` verifies (hashes + both Ed25519 seals), 100 scenes / 414
  artists / DEV-FINAL artist-disjoint, FINAL grown to **88** pairs (min gate
  raised to 80 for schema ≥ 6), records actual ranked lists. Deciding FINAL is
  ListenBrainz session co-listening; training source is independent
  (Music4All/Last.fm) — `same_dataset_or_api=false`.
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET and
  strengthened. Iteration 5 adds a genuinely new mechanism — **query-conditioned
  item2vec collaborative candidate generation + learned reranking over an
  audio∪collaborative∪production union** — exactly the iter-4-recommended axis,
  and documents its rejection alongside the six iter-4 audio encoders.
- [ ] **AC#3 — ≥20% clear gain / no scene −>10% / ≥80% top-5 / no junk.**
  **NOT MET — deciding failure on both sub-gates, honestly reported.**
  (a) Retrieval: winner **−97.1%** vs baseline, 2/88 improved, all 8 gates
  false, CI ⊇ 0, `punk-hardcore` −100%. (b) Coherence: **13/20 (65%) < 80%**.
  The only positive is the leaky unmasked ablation (invalid decider).
- [x] **AC#4 — External validation equivalent-or-better; not same-artist-only.**
  MET as *supporting* evidence. Deezer related-artists: 0.158→0.267,
  `ci95=[0.036, 0.188]` (excludes zero), but explicitly labelled *artist-level
  taste affinity* and **not** used as the decider; `benchmark_artist_overlap=[]`.
- [x] **AC#5 — Wired into canonical + hosted, live-verified.** MET (unchanged,
  correctly). Production **remains iteration-3 `dual_sonic64_guardrail`**;
  `iteration5_deployed=false`, `deployment_attempted=false`,
  `production_unchanged=true`; no serving-path diff; `release_assets_uploaded=
  false`, `manifest_updated=false`, `repository_pushed=false` (local branch is
  ahead by 4, unpushed). Live stats OK (272,853; `2026.07.11-dual-sonic64`).
- [x] **AC#6 — Resources measured, fits limits, no silent fallback.** MET.
  Added runtime **+2.03 MB** (collab index 1.98 MB + scorer 48 KB), RSS delta
  **+48 MB**, warm latency **p50 0.257 s / p95 0.288 s**. Compact enough to
  ship — the Builder withheld it **because quality failed**, not resources
  (explicit non-ship, no hidden downgrade). Unmasked (leaky) asset not shipped.
- [x] **AC#7 — Regression tests + full suite + docs.** MET. Quality gate
  reproduced: **408 passed in 19.02 s** (up from 399); `test_collaborative.py`
  added. `build` passes; `pip_audit` clean (setuptools floor raised to
  ≥78.1.1 to clear PYSEC-2025-49); `weights_only=True`; no secrets. README /
  CASE_STUDY document the negative result honestly ("failed decisively",
  "13/20", "Nothing from iteration 5 was deployed") with full reproduction
  commands — **no misleading "pass" headline** reintroduced.

---

## Integrity audit — iteration-4 issues all resolved

- **`goal.md` RESTORED.** Commit `5f5973f` re-adds the previously-deleted line
  *"Do not add attribution trailers or generated-by taglines."* (+1). The
  immutable spec now matches its frozen form. ✔
- **`RANKINGS_LOCKED` now enforced** (`rankings_locked_before_open=true`,
  `locked_at` precedes the FINAL open). ✔
- **Divide-by-zero `relative_gain` fixed** in `_bootstrap` (returns `None`
  when baseline ≤ 0; on FINAL, baseline 0.008072 > 0 so `−0.971` is real). ✔
- Commit hygiene clean: `[B]` marker present, **no attribution trailer**.

**Residual (non-blocking) note:** direct-edge masking removes *first-order*
user co-occurrences (2236→0) before training, but **transitive** leakage
(A–C–B via a shared third track) is not eliminated by first-order masking.
This does **not** affect the verdict — the masked model is the decider and it
*fails*, so any residual leakage could only have flattered a result that is
already negative. Worth tightening only before a future PASS is claimed.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — 408 passed in 19.02s (independently reproduced).
- Note: the gate confirms code correctness only; it does not adjudicate AC#3,
  which fails on the once-opened FINAL and the 13/20 direct gate.

---

## Issues Found (ranked)
1. **[BLOCKER] AC#3 unmet on both sub-gates.** Deciding masked winner
   **−97.1%** vs baseline (2/88 improved, all 8 gates false, `punk-hardcore`
   −100%); direct coherence **13/20 (65%) < 80%**. Honestly reported
   (`final_pass=false`, `passes_gate=false`).
2. **[STRUCTURAL] 5% catalogue collaborative coverage** (13,680/272,853; only
   39/88 FINAL pairs mappable) caps candidate recall at ~9%@1000 — a reranker
   cannot rank targets that never enter the pool.
3. **[MEASUREMENT] One-positive-per-seed FINAL** keeps the primary metric on a
   noise floor (every CI ⊇ 0). It is a weak proxy for top-5 satisfaction, but
   must be fixed *by strengthening the gold set, not by lowering the bar*.
4. **[COVERAGE] 4/20 held-out difficult seeds absent from the catalogue**
   (cold-start) — unmeasurable as shipped.

## Credit where due
Did the iter-4-recommended experiment properly: an **independent**,
query-conditioned collaborative source (real CC-BY Music4All-Onion, verified),
leakage-masked deciding model with an unmasked diagnostic to expose edge
memorization, candidate-recall reported as a gating metric, all three iter-4
integrity defects fixed, production left untouched, 408 green tests, honest
docs. The remaining gap is scientific, and the Builder does not disguise it.

---

## What Must Be Fixed (to pass) — evidence-driven, do NOT relax the goal

The deciding failure is now clearly **coverage + measurement**, not ranking.

1. **Break the 5% collaborative-coverage ceiling.** Union *multiple*
   independent co-listening sources (playlist co-occurrence, Deezer/Spotify
   related, additional Last.fm windows) to map the **majority** of the served
   272k catalogue, and add a **cold-start bridge** so unmapped seeds inherit
   collaborative candidates via their nearest *mapped* audio neighbours.
   Gate on **candidate-recall@K before reranking** — 9%@1000 means the pool,
   not the reranker, is the bottleneck. Keep the model source independent of
   the ListenBrainz source that builds FINAL, and mask transitive (not just
   first-order) leakage.
2. **Fix the held-out coherence gate directly.** Ingest or replace the 4
   cold-start seeds (glaive, Anri, Coltrane) so the difficult set is fairly
   measurable, and add scene/style-consistency guardrails for positions 1–3
   to repair the 3 genuine misses (Pixies→trip-hop, Post Malone→rap-only,
   JVKE→shoegaze). Target ≥16/20.
3. **Make the primary score higher-dynamic-range WITHOUT lowering the bar.**
   Replace single-counterpart gold with a **multi-positive, graded-relevance**
   gold set per seed (top-N independent co-listening/playlist neighbours,
   deduped of duplicates/karaoke/slowed per the goal's exclusions), scored by
   nDCG@10 / Recall@10. This lifts the metric off the noise floor so a real
   ≥20% gain is detectable with non-degenerate CIs, stays grounded in
   independent human behaviour (goal-permitted), and **retains** the ≥80%
   held-out top-5 coherence gate as the deciding human-aligned check. This is a
   measurement-defect fix, not a post-hoc relaxation.
4. **Only then re-open FINAL once** (rankings locked first, as now enforced)
   and claim the ≥20% gain on the axis the shipped mechanism actually
   addresses — never on a blended noise-floor metric.
