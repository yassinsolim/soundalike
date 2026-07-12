# Inspector Feedback — Iteration 4

## Verdict: FAIL

Independent inspection of commit `00255e8` against the immutable
`goal.md` and `inspector-feedback-3.md`. Every Builder claim was treated
as untrusted and re-derived from the frozen protocol artifacts, the
production index, the committed rankings, and the test suite.

**Bottom line:** This is the most honest and best-engineered iteration by
far — it fixes essentially every *methodological* defect from iterations
1–3 (popularity-prior confound, benchmark source credibility, deployment
provenance, misleading headline, real audio training, tamper-evident
freeze). But the single **deciding** criterion, **AC#3**, is still **not
met**, and the Builder correctly reports this itself: the once-opened
FINAL set fails (`final_pass=false`, `deployed=false`). The immutable goal
is therefore not satisfied. Separately, the Builder committed an
**integrity violation** by editing the immutable `goal.md`.

---

## What I reproduced (verified from frozen artifacts)

**FINAL (40 pairs, once-opened, `open_number=1`)** — recomputed
independently from `protocol-v5/winner-rankings.json`,
`frozen-baseline-rankings.json`, and `final-test-manifest.json` (targets
re-resolved against `ml_data/deepvibe_index_v5.npz`). My numbers match
`final-once-v5.json` **exactly**:

| FINAL method | primary | R@10 | R@20 | R@50 | MRR | pairs found |
|---|---|---|---|---|---|---|
| production_baseline | 0.000000 | 0.000 | 0.000 | 0.000 | 0.000000 | 0/40 |
| iteration3_deployed | 0.000278 | 0.000 | 0.000 | 0.025 | 0.000833 | 1/40 |
| raw_encoder | 0.000000 | 0.000 | 0.000 | 0.000 | 0.000000 | 0/40 |
| **audio_priors_zero** | **0.012004** | **0.025** | 0.025 | 0.025 | 0.003125 | 1/40 (rank 8) |
| **winner** (stratified-tail) | **0.000595** | 0.000 | 0.025 | 0.025 | 0.001786 | 1/40 (rank 14) |

- The winner retrieves **exactly one** of 40 pairs (`FINAL-LB030`:
  Beach Weather "Sex, Drugs, Etc." → girl in red "we fell in love in
  october") at **rank 14**.
- The **plain audio-cosine ablation `audio_priors_zero` beats the
  selected winner** on FINAL (same pair at **rank 8**, primary **20×
  higher**). The elaborate stratified-tail reranker is **net noise on
  unseen data** — a textbook overfit-to-DEV signature.
- Pass gates (from `final-once-v5.json`, reproduced):
  `ci_excludes_zero=false`, `meaningful_count=false` (1 improved pair),
  `probability_positive=0.6364`. **`final_pass=false`.**

**DEV (67 pairs, selection split):** winner `dev_primary=0.025585`,
`dev_relative_gain=+69.9%`, `dev_ci95=[0.000116, 0.028508]`. The DEV gain
is real but **did not generalize** to FINAL. Note DEV and FINAL are
**different evidence types** (DEV = critic/editorial "sounds-like";
FINAL = ListenBrainz co-listening), so DEV is not a clean held-out
estimator of the FINAL task.

**Root-cause diagnosis (decisive).** I checked whether the near-zero
recall is coverage, aliasing, candidate generation, or geometry:
- **All 40 FINAL queries AND all 40 targets match at the *exact*
  title+artist level in the 272,853-row index.** It is **not** catalogue
  availability and **not** alias/version mapping.
- DEV candidate-union recall (from `audio-dev-results-v5.json`): raw
  audio ~**0.13 @1000**; multivector late-interaction **0.217 @1000 /
  0.483 @5000**. Even a 5,000-wide net misses the majority of targets.
- Conclusion: the misses are an **embedding-geometry / target-mismatch**
  problem. Audio nearest-neighbours are simply **not** the co-listening
  neighbours the FINAL benchmark rewards. The reranker cannot fix a
  target that never enters the candidate pool.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / held-out / credible sources.**
  MET and much improved. `soundalike_pairs.v5.json`: 107 pairs, **all
  `category_a_sonic`** (samples/legal/cover/weak categories removed
  entirely — fixes the iter-2/3 category concern). `validate_benchmark`
  passes: 67 dev / 40 final, **85 scenes**, **234 artists**, every pair
  cross-artist, **no artist repeats anywhere**, and **DEV/FINAL are
  artist-disjoint** (`artist_overlap=[]`) → no transitive-component
  leakage possible. Provenance is real: `source-url-audit-v5.json`
  HTTP-checked 190 URLs (186×200, 2×403 paywalled, 2 stale 404s corrected
  to canonical/Wayback). **Caveat:** the *deciding* FINAL split is
  ListenBrainz session-based co-listening, i.e. **behavioural
  co-occurrence, not sonic similarity** — arguably a non-sonic deciding
  signal (see AC#3 root cause).
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET
  (strongest area). Six genuinely GPU-trained audio approaches, all with
  losses/wall-times/CUDA telemetry: FMA cross-artist **SupCon** (617s),
  **BYOL** (390s), **teacher distillation** over 256k rows (319s,
  15,570 benchmark rows excluded), **independent-pair audio metric**
  (3,967 listening pairs, benchmark-artist-disjoint), **multivector
  late-interaction**, and the **stratified-tail** winner. Full 272,853×128
  f16 catalog embeddings built and hash-verified for distill/supcon/byol.
  This decisively answers iteration-3 Blocker #2 ("the gain was a
  popularity prior, not audio").
- [ ] **AC#3 — ≥20% clear gain / no scene −>10% / ≥80% top-5 / no junk.**
  **NOT MET (deciding failure, honestly reported).** FINAL fails on the
  merits above: 1/40 retrieval, CI includes zero, only one improved pair,
  and the winner is beaten by a plain-audio ablation. `passes.relative_gain`
  in the artifact is a **divide-by-zero artifact** (baseline primary is
  literally 0.0, so `relative_gain=5.95e8`), not a real +20%. The Builder
  does **not** claim a pass.
- [x] **AC#4 — External validation equivalent-or-better.** MET/neutral.
  External sources are not model features; benchmark-artist overlap empty.
- [x] **AC#5 — Winner wired into canonical + hosted, live-verified.**
  MET (unchanged from iter-3, correctly). Production **remains the
  iteration-3 `dual_sonic64_guardrail`** (index v5 still carries
  sonic/clap/wiki fields; iteration-4 assets **not** uploaded because
  FINAL failed). `live-retained-production-v5.json`: **12** diverse seeds,
  all search/recommend/method/version/library/previews OK. Deployment
  provenance **corrected** to `source_sha=2366b7c` with an explicit note
  retracting the earlier bogus "PR #30 merge" (fixes iter-3 issue #4).
- [x] **AC#6 — Resources measured, fits limits, no silent fallback.**
  MET (honest). Research method = **508.8 MB** disk (299 MB index +
  209.6 MB aux) and **~2.85 GB RSS** after load. That RSS is **at/over
  the practical Vercel serverless memory ceiling and would be risky to
  deploy** — so the Builder **did not deploy it** (`deployed=false`,
  `production.unchanged=true`). This is exactly the "must fit … without
  silent quality fallbacks" behaviour the goal asks for: an explicit
  non-ship, not a hidden downgrade.
- [x] **AC#7 — Regression tests + full suite + docs.** MET. Quality gate
  reproduced: **399 passed in 18.86s** (up from 308). `build` passes,
  `pip_audit` clean (0 vulns), torch checkpoints hardened to
  `weights_only=True`. Quality filter on the real index:
  **0 obvious false-negatives, 0 curated false-positives** (1,361 removed
  of 272,853; legit "Karaoke"/"Cover Me" titles retained). The misleading
  **+88.3%** headline is **gone** from README and now appears in
  CASE_STUDY only in a corrective disclosure (fixes iter-3 issue #5).

---

## Protocol / integrity audit

- **Freeze is real and verifies.** All frozen hashes in
  `protocol-v5/state.json` match on disk (benchmark, index, manifest,
  baseline, method-manifest, dev-report, winner-rankings, and all 4
  method assets). The SHA-256 canonical integrity signature verifies, and
  the **detached Ed25519 seal verifies via `ssh-keygen -Y verify`**.
  `final_open_count=1` — opened exactly once. Winner rankings are
  `target_labels_compared=false` and their content hash matches.
- **[MAJOR, self-disclosed] RANKINGS_LOCKED transition was NOT enforced
  for this run.** `rankings_lock_transition_enforced=false`; there is no
  `rankings_locked_at`. The winner rankings were generated (08:46:23)
  before the final was opened (08:46:37) and their hash is recorded, but
  the run **bypassed** the `commit_rankings` gate that would
  cryptographically bind rankings before scoring. The code was hardened
  afterward. **Mitigation:** because `final_pass=false`, there was no
  incentive or benefit to regenerate rankings post-hoc — this weakens a
  *pass* claim, not a *fail*. Acceptable for a negative result; **must be
  enforced before any future PASS is trusted.**
- **[MAJOR] Immutable `goal.md` was modified by the Builder.** Commit
  `00255e8` deletes the line *"Do not add attribution trailers or
  generated-by taglines."* from `goal.md` (present at `2366b7c` and at
  inspector commit `9453589`). Editing the immutable goal is a governance
  violation regardless of intent. The commit message itself is clean, so
  there is no *functional* harm this iteration, but **`goal.md` must be
  restored to its frozen form** and the immutable spec must not be touched
  again.
- Commit hygiene otherwise clean: message `feat(recs): [B] train robust
  audio soundalike retrieval` has the `[B]` marker and **no attribution
  trailer**.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** (399 passed in 18.86s)
- Note: the gate confirms code correctness only; it does not adjudicate
  AC#3, which fails on the once-opened FINAL evidence.

---

## Issues Found (ranked)
1. **[BLOCKER] AC#3 not met.** Once-opened FINAL fails: winner retrieves
   1/40 (rank 14), CI ⊇ 0, one improved pair, and is **beaten by the
   plain-audio ablation** (rank 8). Honestly reported (`final_pass=false`)
   but the deciding criterion is objectively unmet.
2. **[MAJOR] Immutable `goal.md` edited** (attribution-prohibition line
   removed). Restore it.
3. **[MAJOR] Freeze protocol gap** (`RANKINGS_LOCKED` skipped this run);
   tolerable only because the result is a disclosed FAIL.
4. **[DESIGN] Benchmark/method mismatch.** The deciding FINAL truth is
   co-listening (collaborative) while the method is constrained
   audio-only; DEV is critic-sonic. The three measure different things,
   which floors the deciding metric (R@5=0 for every method).

## Credit where due (real progress over iters 1–3)
Popularity-prior confound removed (method genuinely audio-only, verified
in `_rank_audio_priors_zero`); six real GPU-trained audio encoders; full
272k index rebuilt; tamper-evident hash+Ed25519 freeze; category-clean,
artist-isolated, HTTP-verified benchmark; corrected deployment provenance;
removed misleading headline; production left untouched rather than shipping
a failing/oversized method; 399 green tests. The remaining gap is
scientific, and — importantly — the Builder no longer disguises it.

---

## What Must Be Fixed (to pass) — evidence-driven, not "train more"

The misses are **geometry/target-mismatch**, not coverage (every FINAL
target is in-catalog yet never enters the audio top-1000). Rank/tail
tuning is rearranging an empty pool. Concretely:

1. **Fix candidate generation before any reranking.** Union audio-NN
   candidates with **collaborative/co-listening-NN candidates** (per-seed
   playlist/co-occurrence neighbours), then rerank the union. This
   directly attacks the observed ceiling (candidate recall ~0.13 @1000,
   0.48 @5000). Report candidate-recall@K as a gating metric — a reranker
   cannot rank a target it never sees.
2. **Add a *query-conditioned* collaborative prior — which the goal
   explicitly permits (AC#2).** Iteration-3's rejected confound was a
   *static, seed-independent global popularity* prior; a **per-seed**
   co-listening/graph reranker is a different, in-scope mechanism and is
   exactly what recovers these pairs. **Avoid circularity:** the model's
   collaborative source must be **independent of the source that built the
   benchmark** (FINAL is ListenBrainz session-based; draw the model prior
   from Last.fm similar-tracks, Deezer/Spotify playlist co-occurrence, or
   a *different* ListenBrainz algorithm) — otherwise it is train-on-test
   leakage.
3. **Align the deciding metric with the shipped mechanism.** Either
   evaluate the audio-only method against **sonic** ground truth
   (critic/participant "sounds-like", which the DEV split already
   contains) so it is measured on what it optimizes, **or** commit to the
   co-listening target and ship the query-conditioned collaborative
   retrieval above. Measure sonic-recall and collaborative-recall as
   **separate axes** and claim the ≥20% gain only on the axis the shipped
   method addresses — a single blended noise-floor metric (R@5=0
   everywhere) can never demonstrate a "clear" improvement.
4. **Restore `goal.md`** and **enforce the `RANKINGS_LOCKED` transition**
   before the next FINAL open, so any future PASS is cryptographically
   defensible.
